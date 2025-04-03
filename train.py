import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import numpy as np
from sklearn.metrics import classification_report, f1_score
import torch
from torch import nn
from transformers import BertModel
import os

# 1. Veri yükleme (orijinal TSV dosyası)
df = pd.read_csv("./dataset/troff-v1.0.tsv", sep="\t", header=None, names=["text", "label"])

# 2. Etiketlerin işlenmesi
# Çoklu etiketleri işleyebilmek için etiketleri ayırma
df['labels_list'] = df['label'].str.split()

# Sadece tek başına X etiketli olan tweetleri çıkar (Türkçe olmayan veya anlaşılamayan içerik)
df = df[~((df['labels_list'].apply(len) == 1) & (df['labels_list'].apply(lambda x: x[0] == 'X')))]

# Etiket kümesini oluştur
labels = ["non", "prof", "grp", "ind", "oth"]
label_dict = {label: i for i, label in enumerate(labels)}

# Çoklu etiket matrisi oluşturma (multi-label classification için)
df['label_matrix'] = df['labels_list'].apply(
    lambda x: [1 if label in x and label != 'X' else 0 for label in labels]
)

# Zorluk etiketi (X ikincil etiket olarak kullanıldığında)
df['is_difficult'] = df['labels_list'].apply(lambda x: 1 if 'X' in x else 0)

# 3. Veriyi hiyerarşik yapı için düzenleme
# level 1: offensive/non-offensive (non etiketi varsa 0, yoksa 1)
df['offensive'] = df['label_matrix'].apply(lambda x: 0 if x[0] == 1 else 1)

# level 2: targeted/untargeted (prof etiketi varsa 0, bir hedef varsa 1)
df['targeted'] = df['label_matrix'].apply(lambda x: 0 if x[1] == 1 and sum(x[2:]) == 0 else 
                                           1 if sum(x[2:]) > 0 else 0)

# level 3: target type (grp: 0, ind: 1, oth: 2, multiple: 3)
def get_target_type(labels):
    has_grp = labels[2] == 1
    has_ind = labels[3] == 1
    has_oth = labels[4] == 1
    
    if sum([has_grp, has_ind, has_oth]) > 1:
        return 3  # multiple targets
    elif has_grp:
        return 0  # group
    elif has_ind:
        return 1  # individual
    elif has_oth:
        return 2  # other
    else:
        return -1  # no target (should not happen for targeted=1)

df['target_type'] = df['label_matrix'].apply(get_target_type)
df.loc[df['targeted'] == 0, 'target_type'] = -1  # Hedefsiz olanlar için -1

# 4. Eğitim/test bölünmesi
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['offensive'], random_state=42)
train_ds = Dataset.from_pandas(train_df)
test_ds = Dataset.from_pandas(test_df)

# 5. Özel model tanımlama - Hiyerarşik sınıflandırma için
class HierarchicalOffensiveClassifier(nn.Module):
    def __init__(self, model_name, num_labels=5):
        super(HierarchicalOffensiveClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.num_labels = num_labels
        
        # Hiyerarşik sınıflandırıcılar
        self.offensive_classifier = nn.Linear(self.bert.config.hidden_size, 2)  # offensive or not
        self.targeted_classifier = nn.Linear(self.bert.config.hidden_size, 2)   # targeted or not
        self.target_type_classifier = nn.Linear(self.bert.config.hidden_size, 4)  # grp, ind, oth, multiple
        
        # Çoklu etiket sınıflandırıcı
        self.multi_label_classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        
        # Zorluk tahmini (X etiketi için)
        self.difficulty_classifier = nn.Linear(self.bert.config.hidden_size, 2)
        
        # Tüm tensörleri bitişik yap
        self._make_tensors_contiguous()
    
    def _make_tensors_contiguous(self):
        """Tüm model parametrelerini bitişik hale getir"""
        for name, param in self.named_parameters():
            if not param.is_contiguous():
                param.data = param.data.contiguous()
    
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        sequence_output = outputs.last_hidden_state
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        
        # Hiyerarşik sınıflandırma çıktıları
        offensive_logits = self.offensive_classifier(pooled_output)
        targeted_logits = self.targeted_classifier(pooled_output)
        target_type_logits = self.target_type_classifier(pooled_output)
        
        # Çoklu etiket sınıflandırma çıktısı
        multi_label_logits = self.multi_label_classifier(pooled_output)
        
        # Zorluk çıktısı
        difficulty_logits = self.difficulty_classifier(pooled_output)
        
        return {
            'offensive_logits': offensive_logits,
            'targeted_logits': targeted_logits,
            'target_type_logits': target_type_logits,
            'multi_label_logits': multi_label_logits,
            'difficulty_logits': difficulty_logits
        }

# 6. Model ve tokenizer (Türkçe BERT)
model_name = "dbmdz/bert-base-turkish-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 7. Yeni veri hazırlama yaklaşımı
# Önce tokenize işlemi yapılır, sonra etiketler eklenir

# Her bir batchteki örnekleri tokenize et
def tokenize_function(examples):
    return tokenizer(
        examples["text"], 
        padding="max_length", 
        truncation=True, 
        max_length=128,
        return_tensors=None  # Henüz tensöre dönüştürmeyin
    )

# Tokenize işlemini uygula
train_ds = train_ds.map(tokenize_function, batched=True)
test_ds = test_ds.map(tokenize_function, batched=True)

# Şimdi etiketleri hazırla
def prepare_labels(examples):
    batch_size = len(examples["text"])
    
    # Etiketleri numpy dizilerine dönüştür (tensöre dönüştürmek yerine)
    examples["labels"] = np.array(examples["label_matrix"], dtype=np.float32)
    examples["offensive"] = np.array(examples["offensive"], dtype=np.int64)
    examples["targeted"] = np.array(examples["targeted"], dtype=np.int64)
    examples["target_type"] = np.array(examples["target_type"], dtype=np.int64)
    examples["is_difficult"] = np.array(examples["is_difficult"], dtype=np.int64)
    
    return examples

# Etiketleri hazırla
train_ds = train_ds.map(prepare_labels, batched=True)
test_ds = test_ds.map(prepare_labels, batched=True)

# Gereksiz sütunları kaldır
columns_to_remove = ["label", "labels_list", "label_matrix"]
train_ds = train_ds.remove_columns(columns_to_remove)
test_ds = test_ds.remove_columns(columns_to_remove)

# Veri seti formatlarını ayarla
train_ds.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "token_type_ids", "labels", "offensive", "targeted", "target_type", "is_difficult"]
)
test_ds.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "token_type_ids", "labels", "offensive", "targeted", "target_type", "is_difficult"]
)

# Transformers için özel sınıflandırıcı modeli
model = HierarchicalOffensiveClassifier(model_name, num_labels=len(labels))

# 8. Özel eğitim döngüsü (Trainer sınıfını özelleştirerek)
class OffensiveTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Labels ve diğer hedef değerleri çıkar
        labels = inputs.pop("labels", None)
        offensive_labels = inputs.pop("offensive", None)
        targeted_labels = inputs.pop("targeted", None)
        target_type_labels = inputs.pop("target_type", None)
        is_difficult = inputs.pop("is_difficult", None)
        
        # Model çıktılarını al
        outputs = model(**inputs)
        
        # Kayıp fonksiyonları
        loss_fct_binary = nn.CrossEntropyLoss()
        loss_fct_multi = nn.BCEWithLogitsLoss()
        
        # Hiyerarşik sınıflandırma kayıpları
        offensive_loss = loss_fct_binary(outputs['offensive_logits'], offensive_labels)
        
        # Sadece offensive olan örnekler için targeted loss hesapla
        targeted_mask = (offensive_labels == 1)
        if targeted_mask.sum() > 0:
            targeted_loss = loss_fct_binary(
                outputs['targeted_logits'][targeted_mask], 
                targeted_labels[targeted_mask]
            )
        else:
            targeted_loss = torch.tensor(0.0, device=offensive_labels.device)
        
        # Sadece targeted olan örnekler için target_type loss hesapla
        target_type_mask = (targeted_labels == 1) & (offensive_labels == 1)
        if target_type_mask.sum() > 0:
            target_type_loss = loss_fct_binary(
                outputs['target_type_logits'][target_type_mask], 
                target_type_labels[target_type_mask]
            )
        else:
            target_type_loss = torch.tensor(0.0, device=offensive_labels.device)
        
        # Çoklu etiket sınıflandırma kaybı
        multi_label_loss = loss_fct_multi(outputs['multi_label_logits'], labels)
        
        # Zorluk tahmini kaybı
        difficulty_loss = loss_fct_binary(outputs['difficulty_logits'], is_difficult)
        
        # Toplam kayıp
        loss = offensive_loss + targeted_loss + target_type_loss + multi_label_loss + difficulty_loss
        
        return (loss, outputs) if return_outputs else loss
    
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        
        # Labels ve diğer hedef değerleri çıkar
        labels = inputs.pop("labels", None)
        offensive_labels = inputs.pop("offensive", None)
        targeted_labels = inputs.pop("targeted", None)
        target_type_labels = inputs.pop("target_type", None)
        is_difficult = inputs.pop("is_difficult", None)
        
        with torch.no_grad():
            outputs = model(**inputs)
            
            # Tahminleri al
            offensive_preds = torch.argmax(outputs['offensive_logits'], dim=-1)
            targeted_preds = torch.argmax(outputs['targeted_logits'], dim=-1)
            target_type_preds = torch.argmax(outputs['target_type_logits'], dim=-1)
            multi_label_preds = torch.sigmoid(outputs['multi_label_logits'])
            difficulty_preds = torch.argmax(outputs['difficulty_logits'], dim=-1)
        
        # Tahminleri sonuç sözlüğünde topla
        predictions = {
            'offensive_preds': offensive_preds,
            'targeted_preds': targeted_preds,
            'target_type_preds': target_type_preds,
            'multi_label_preds': multi_label_preds,
            'difficulty_preds': difficulty_preds
        }
        
        # Etiketleri sonuç sözlüğünde topla
        label_dict = {
            'offensive_labels': offensive_labels,
            'targeted_labels': targeted_labels,
            'target_type_labels': target_type_labels,
            'labels': labels,
            'is_difficult': is_difficult
        }
        
        return (None, predictions, label_dict)
    
    def _save(self, output_dir: str, state_dict=None):
        """Özelleştirilmiş kaydetme metodu, tüm tensörlerin bitişik olmasını sağlar"""
        # Eğer state_dict verilmemişse, modelden al
        if state_dict is None:
            state_dict = self.model.state_dict()
        
        # Tüm tensörlerin bitişik olmasını sağla
        for key in state_dict:
            if isinstance(state_dict[key], torch.Tensor) and not state_dict[key].is_contiguous():
                state_dict[key] = state_dict[key].contiguous()
        
        # Kaydetme işlemi için üst sınıfın metodunu çağır
        # Ancak PyTorch save metodunu kullan, safetensors yerine
        os.makedirs(output_dir, exist_ok=True)
        torch.save(state_dict, os.path.join(output_dir, "pytorch_model.bin"))
        
        # Konfigürasyon dosyasını kaydet
        if hasattr(self.model, "config") and self.model.config is not None:
            self.model.config.save_pretrained(output_dir)
        
        # Özel model için BERT yapılandırmasını da kaydet
        if hasattr(self.model, "bert") and hasattr(self.model.bert, "config"):
            self.model.bert.config.save_pretrained(output_dir)

# 9. Metrik hesaplama
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    
    # Çoklu etiket tahminleri
    # Prediction dictionary'den doğrudan tensörü alarak işlem yapın
    multi_preds = predictions['multi_label_preds']
    if isinstance(multi_preds, torch.Tensor):
        multi_preds = multi_preds.cpu().numpy()
    
    # Binary etiketlere dönüştür (eşik: 0.5)
    multi_preds = (multi_preds > 0.5).astype(np.int32)
    
    # Label tensörlerini numpy array'lerine dönüştür
    label_array = labels['labels']
    if isinstance(label_array, torch.Tensor):
        label_array = label_array.cpu().numpy()
    
    # F1 skorları
    macro_f1 = f1_score(label_array, multi_preds, average='macro', zero_division=0)
    weighted_f1 = f1_score(label_array, multi_preds, average='weighted', zero_division=0)
    
    # Hiyerarşik tahminler için metrikler
    offensive_preds = predictions['offensive_preds']
    offensive_labels = labels['offensive_labels']
    
    if isinstance(offensive_preds, torch.Tensor):
        offensive_preds = offensive_preds.cpu().numpy()
    
    if isinstance(offensive_labels, torch.Tensor):
        offensive_labels = offensive_labels.cpu().numpy()
        
    offensive_acc = (offensive_preds == offensive_labels).mean()
    
    # Sonuçları raporla
    return {
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "offensive_acc": offensive_acc
    }

# 10. Eğitim ayarları
training_args = TrainingArguments(
    output_dir="./offensive_model_hierarchical",
    eval_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=4,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    # Tensör dönüşüm hatalarını önlemek için
    dataloader_drop_last=True,
    remove_unused_columns=False,  # Özel model için gerekli
    report_to=["tensorboard"]  # TensorBoard desteği ekle
)

# 11. Eğitici ve eğitim başlat
trainer = OffensiveTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    compute_metrics=compute_metrics,
    # Özel bir veri koleksiyonlayıcısı eklemek için:
    data_collator=lambda data: {
        'input_ids': torch.stack([f['input_ids'] for f in data]),
        'attention_mask': torch.stack([f['attention_mask'] for f in data]),
        'token_type_ids': torch.stack([f['token_type_ids'] for f in data]),
        'labels': torch.stack([f['labels'] for f in data]),
        'offensive': torch.stack([f['offensive'] for f in data]),
        'targeted': torch.stack([f['targeted'] for f in data]),
        'target_type': torch.stack([f['target_type'] for f in data]),
        'is_difficult': torch.stack([f['is_difficult'] for f in data])
    }
)

trainer.train()

# 12. Model ve tokenizer kaydet
trainer.save_model("offensive_model_hierarchical")
tokenizer.save_pretrained("offensive_model_hierarchical")
