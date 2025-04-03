# Turkish Offensive Tweet Corpus: Guidelines (en)
**This page contains samples of offensive/inappropriate language in Turkish.**

This page (guidelines in English) is for reference. Annotators were instructed to follow the [guidelines in Turkish](guidelines-tr).

Thank you for helping us out for annotating this Twitter data set to identify offensive statements. The definitions of offensive language and its types are not necessarily clear-cut, and opinions differ to some extent. Nevertheless, we would like to get as unified annotations as possible.

Before starting please read the following guidelines carefully. [A short reference](guidelines-short) is provided in the annotations system (click on the book icon on the upper left corner).

Label set
---------

We classify the tweets with one or more or the following labels:

*   **non**offensive: The texts that are clearly not offensive should be marked with this label, e.g., example (9) below.
    
*   **prof**anity, or “untargeted offense”, is use of offensive language without a particular target. This is typically use of swear words, or other “inappropriate” language for stylistic reasons. This also includes jokes or friendly teasing. Examples 6-8 below should be marked **prof**. Profanity also includes “pornographic language” observed in tweets seeking sex partners (or advertisements of prostitution) and others interacting with them.
    
*   **grp** (offense towards a group) means the author intends to offend a group that forms a “unity” based on gender, ethnicity, political affiliation, religious belief or similar aspects of a person’s identity. An offense towards a group or individual that does not share a common identity does not qualify. For example, example (1) and (2) below are towards a clearly defined group (skin color and country/ethnicity), while the offenses in (3) and (4) do not qualify although multiple people are the target of the offense. One of the common but difficult cases in the data is the soccer team fans, which we also consider as a group. Another common issue is the use of racist or sexist expressions in language without a clear intention to insult a particular target as in (16) below. We also consider such expressions as an offense towards a group. A variation is to use of such an expression with the aim of insulting a person as in (17), which should be marked as both **grp** and **ind** (see definition of **ind** below).
    
*   **ind** (offensive towards an individual) is used for offensive targeted to an individual **or group of individuals** that are not related in well-defined manner.
    
*   **oth** target of the offense is not one of the two cases above. This typically include organizations, events etc.
    
*   **X** label is used for two purposes.
    
    *   When used alone, it marks the tweet for exclusion from the data set. We exclude occasional non-Turkish tweets, and when there isn’t enough linguistic material to make a decision, typically spam tweets made of non-sense sequences of words such as (15) below. Do not exclude tweets just because they are not grammatical or well formed. As long as a native speaker can understand the tweet, it should be included.
    *   As a second label for tweets are those that are difficult to decide. Decision of offensiveness for some of the tweets require more context than a single tweet, and sometimes use of figurative language, irony, may make the decision difficult even for humans. This should also cover when it is difficult to decide whether an offensive tweet targets a person or a group. You are are still requested to make a decision. However, adding this label signals that the classification is not straightforward.

When appropriate, you may chose multiple labels. However, you should try to pick the smallest set of applicable labels. For example, a tweet may contain both an offense to an individual and a discriminating/hate attitude toward a group as in example (10) below. In general, a non-offensive tweet should not have any of the other labels above.

We also mark the use of self-censored or contracted words like examples (13) and (14) below as **prof**anity or offensive as if the respective words were used in their full form.

The annotation process
----------------------

We describe the labels above in a “flat” manner. However, the annotation process we follow is hierarchical. The following QA pairs give a more flowchart-like procedure to follow

1.  _Is the tweet in Turkish and understandable?_
    *   No: mark tweet **X** for exclusion, and go to next tweet
    *   Yes: continue to step 2
2.  _Is the tweet include offensive/inappropriate language?_
    *   No: mark the tweet **non** go to step 4
    *   Yes: continue to step 3
3.  _Is the offense in the tweet targeted?_
    *   No: mark the tweet **prof** go to step 4
    *   Yes: chose one (or more) of **grp**, **ind**, _\*oth_ based on the definitions above. Please try to limit the number of labels unless it is clear that the tweet includes offense against multiple categories.
4.  _Was the labeling decision difficult_ (precise answer needs more context, tweets includes irony, or for another reason)?
    *   No: go to next tweet
    *   Yes: add the label **X**, go to next tweet

Examples
--------

```
    (1) Amına kodumun zencisi ben de senin koca götünü sikeyim piç
        orospu çocuğu
    (2) İSTÖ İsrail Siyonist terör örgütü tüm müslümanların yok etmesi 
        gereken insanlık düşmanı bir terör örgütüdür...
    (3) Ahlak, namus deyince sadece kadından konuşan herkes ahlaksız
        ve namussuzdur. 
    (4) Lan liseliler yolda yürürken telefona değil yola bakın. üç beş
        tanenizi ibrati alem için tokat manyağı yaparım
    (5) Böyle devam et seni gerizekalı
    (6) Mazlatası olan acımasın buna bassın mazlatayı
    (7) Sensiz uyandığım her günün sabahını sikeyim
    (8) Komple yicem şimdi ikisini de gidip götüme falan sokucam
    (9) teyze olurken bile heyecandan ölüyosam bn doğum falan yapamam
        heralde a dostlar
   (10) Selcuk Dereli @selcuk_dereli Ulann siyasete ne zaman atildin
        diyecemde,zaten soyunuza sopunuza atilmis rum tohumu var.Serefsiz
        a.q cocugu Selcuk Dereli.
   (11) Bugün sokağa davet eden ve sokakta olan herkes ile 
        hesaplaşacağız... ELBET BİRGÜN !
   (12) Allah size fırsat vermesin inşallah
   (13) Yeni sürecinde bunların da ta...
   (14) Akpliler ortalığın a.ina koyuyor kimse ses çıkarmıyor.
   (15) @TalipErdal davranıldığı bedelini nedenlerle Değer verir
        gerektiğinden baktığı şey Söylediğiniz dokunur duygusuz Ceplerime
   (16) Yine gavura sorar gibi sorulan sorular, yine gırık galpleerr ,
        ağlayan yüzler
   (17) @tgmcelebi Sen şimdi vekilsin ve bu yapılanı tavsip ediyorsun
        öylemi amk.. Piçi demokratik tepki ha amk piçi sizin ben amk
        gavur ermeniler amk

```
