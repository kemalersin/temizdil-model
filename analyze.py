import argparse
import torch
from torch import nn
from transformers import AutoTokenizer, BertModel
import numpy as np

# Model sınıfını tanımla (train.py'daki ile aynı olmalı)
class HierarchicalOffensiveClassifier(nn.Module):
    def __init__(self, model_name, num_labels=5, vocab_size=None):
        super(HierarchicalOffensiveClassifier, self).__init__()
        
        # vocab_size parametresi mevcutsa ve bu bir string ise (model yolu), tokenizer'ı yükleyip kelime dağarcığı boyutunu alalım
        if isinstance(model_name, str) and vocab_size is None:
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                vocab_size = len(tokenizer)
                print(f"Tokenizer kelime dağarcığı boyutu: {vocab_size}")
            except Exception as e:
                print(f"Tokenizer yüklenemedi, varsayılan BERT kelime dağarcığı boyutu kullanılacak: {e}")
                vocab_size = None
        
        # BERT modelini yükle, vocab_size varsa kullan
        config_kwargs = {}
        if vocab_size is not None:
            config_kwargs['vocab_size'] = vocab_size
            
        self.bert = BertModel.from_pretrained(model_name, **config_kwargs)
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
    
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        
        # Çıktılar
        offensive_logits = self.offensive_classifier(pooled_output)
        targeted_logits = self.targeted_classifier(pooled_output)
        target_type_logits = self.target_type_classifier(pooled_output)
        multi_label_logits = self.multi_label_classifier(pooled_output)
        difficulty_logits = self.difficulty_classifier(pooled_output)
        
        return {
            'offensive_logits': offensive_logits,
            'targeted_logits': targeted_logits,
            'target_type_logits': target_type_logits,
            'multi_label_logits': multi_label_logits,
            'difficulty_logits': difficulty_logits
        }

def predict_offensive_content(model, tokenizer, text):
    """Metinin saldırgan içeriğini tahmin eder"""
    # Metni tokenize et
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    
    # Tahmin yap
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Hiyerarşik tahminler
    offensive_pred = torch.argmax(outputs['offensive_logits'], dim=1).item()
    targeted_pred = torch.argmax(outputs['targeted_logits'], dim=1).item()
    target_type_pred = torch.argmax(outputs['target_type_logits'], dim=1).item()
    
    # Çoklu etiket tahminleri
    multi_label_probs = torch.sigmoid(outputs['multi_label_logits']).squeeze().tolist()
    multi_label_preds = [1 if prob > 0.5 else 0 for prob in multi_label_probs]
    difficulty_pred = torch.argmax(outputs['difficulty_logits'], dim=1).item()
    
    return {
        'offensive_pred': offensive_pred,
        'targeted_pred': targeted_pred,
        'target_type_pred': target_type_pred,
        'multi_label_probs': multi_label_probs,
        'multi_label_preds': multi_label_preds,
        'difficulty_pred': difficulty_pred
    }

def interpret_predictions(predictions, labels):
    """Tahminleri okunabilir biçimde yorumlar"""
    # Saldırgan içerik var mı?
    is_offensive = predictions['offensive_pred'] == 1
    
    # Etiketler
    predicted_labels = [labels[i] for i in range(len(labels)) if predictions['multi_label_preds'][i] == 1]
    
    # Hedef tipleri
    target_types = ["grup", "birey", "diğer", "çoklu hedef"]
    
    # Sonuçları oluştur
    results = {
        "metin_saldirgan_mi": "Evet" if is_offensive else "Hayır",
        "tahmin_edilen_etiketler": predicted_labels,
        "etiket_olasılıkları": {labels[i]: f"{predictions['multi_label_probs'][i]:.4f}" for i in range(len(labels))},
        "karar_vermesi_zor_mu": "Evet" if predictions['difficulty_pred'] == 1 else "Hayır"
    }
    
    if is_offensive:
        results["hedefli_mi"] = "Evet" if predictions['targeted_pred'] == 1 else "Hayır"
        if predictions['targeted_pred'] == 1:
            results["hedef_turu"] = target_types[predictions['target_type_pred']]
    
    return results

def print_results(results, text):
    """Sonuçları düzgün bir şekilde yazdırır"""
    print("\n" + "="*50)
    print(f"METİN: {text}")
    print("="*50)
    print(f"SALDIRGAN MI: {results['metin_saldirgan_mi']}")
    
    print("\nETİKETLER:")
    if results['tahmin_edilen_etiketler']:
        for label in results['tahmin_edilen_etiketler']:
            print(f"  - {label}")
    else:
        print("  - Etiket bulunamadı")
    
    print("\nETİKET OLASILIKLARI:")
    for label, prob in results['etiket_olasılıkları'].items():
        print(f"  - {label}: {prob}")
    
    if results['metin_saldirgan_mi'] == "Evet":
        print(f"\nHEDEFLİ Mİ: {results['hedefli_mi']}")
        if results['hedefli_mi'] == "Evet":
            print(f"HEDEF TÜRÜ: {results['hedef_turu']}")
    
    print(f"\nKARAR VERMESİ ZOR MU: {results['karar_vermesi_zor_mu']}")
    print("="*50 + "\n")

def main():
    # Argüman ayrıştırıcı
    parser = argparse.ArgumentParser(description="Türkçe metinlerde saldırgan içerik sınıflandırması")
    parser.add_argument("--text", type=str, help="Sınıflandırılacak metin")
    parser.add_argument("--file", type=str, help="Metin dosyası (her satır bir örnek)")
    parser.add_argument("--model_path", type=str, default="./offensive_model_hierarchical", 
                        help="Eğitilmiş model klasörü")
    args = parser.parse_args()
    
    # Model ve tokenizer yükle
    model_path = args.model_path
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Etiketler
    labels = ["non", "prof", "grp", "ind", "oth"]
    
    # Model yükle
    model = HierarchicalOffensiveClassifier(model_path, num_labels=len(labels))
    
    # Model ağırlıklarını yükle (CPU'ya taşı)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(f"{model_path}/pytorch_model.bin", map_location=device))
    model.to(device)
    
    # Metin veya dosya analizi
    if args.text:
        # Tek bir metin analiz et
        predictions = predict_offensive_content(model, tokenizer, args.text)
        results = interpret_predictions(predictions, labels)
        print_results(results, args.text)
    
    elif args.file:
        # Dosyadaki her satırı analiz et
        with open(args.file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                    
                predictions = predict_offensive_content(model, tokenizer, line)
                results = interpret_predictions(predictions, labels)
                print_results(results, line)
    
    else:
        # Komut satırından etkileşimli giriş al
        print("Saldırgan içerik analizi için metin girin (çıkış için 'q' yazın):")
        while True:
            text = input("\nMetin: ")
            if text.lower() == 'q':
                break
                
            predictions = predict_offensive_content(model, tokenizer, text)
            results = interpret_predictions(predictions, labels)
            print_results(results, text)

if __name__ == "__main__":
    main()