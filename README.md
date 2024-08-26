# Red_Wine
Exploratory data analysis &amp; classification work with machine learning model
![Red Wine](https://github.com/user-attachments/assets/71fd7887-e5fd-4fbc-a172-c2f3a7f34f74)
   <a id='top'></a>
<div class="list-group" id="list-tab" role="tablist">
<p style="background-color:#641811;font-family:sans-serif;color:#FFF9ED;font-size:200%;text-align:center;border-radius:9px 9px;">TABLE of CONTENTS</p>   
    
* [1. INTRODUCTION](#1)
    
* [2. IMPORTING NECESSARY LIBRARIES](#2)
    
* [3. LOADING DATASET](#3)
    
* [4. INITIAL INFORMATION ABOUT DATASET](#4)
    
    * [4.1. Get Initial Information](#4.1)    
    * [4.2. Descriptive Statistics of Numeric Variables](#4.2)    
    * [4.3. Pandas Profiling](#4.3)    
    * [4.4. Check null Values](#4.4)
    * [4.5. Rename Column Names](#4.5) 
    
    
* [5. DATA VISUALIZATION](#5)
    
    * [5.1. Histplot](#5.1)    
    * [5.2. Pairplot](#5.2)    
    * [5.3. Scatterplot](#5.3)    
    * [5.4. Smooth Kernel Density with Marginal Histograms](#5.4)
    * [5.5. Regplot](#5.5)
    * [5.6. Hexagonal Binned Plot](#5.6)
    * [5.7. Visualization with Plotly Express](#5.7)
    * [5.8. Heatmap and Correlation](#5.8)
    
    
    
* [6. DATASET PREPROCESSİNG](#6)
    
    * [6.1. Look at Dataset](#6.1)    
    * [6.2. Divide Quality Range into 2 Parts](#6.2)    
    * [6.3. Look at Dataset (with changed 'quality' variable)](#6.3)    
    * [6.4. Select Dependent and Independent Variables](#6.4)
    * [6.5. Split Dataset into Train and Test Sets](#6.5)
    * [6.6. Standardization](#6.6)    

    
    
* [7. BUİLDİNG CLASSİFİCATİON MODELS](#7)
    
    * [7.1. K-Nearest Neighbors (KNN) Model](#7.1)    
    * [7.2. Hyperparameter Tuning for KNN Model](#7.2)    
    * [7.3. Get Best Parameters of KNN Model](#7.3)    
    * [7.4. Build KNN Model with Best Parameters](#7.4)
    * [7.5. Accuracy Score of KNN Model on Test set](#7.5)
    * [7.6. Classification Report of KNN Model](#7.6)    
    * [7.7. Gradient Boosting Machines (GBM) Model](#7.7)     
    * [7.8. Hyperparameter Tuning for GBM Model](#7.8)    
    * [7.9. Get Best Parameters of GBM Model](#7.9)    
    * [7.10. Build GBM Model with Best Parameters](#7.10)   
    * [7.11. Accuracy Score of GBM Model on Test set](#7.11)    
    * [7.12. Classification Report of GBM Model](#7.12)    
    * [7.13. Light GBM Model](#7.13)    
    * [7.14. Hyperparameter Tuning for Light GBM Model](#7.14)    
    * [7.15. Get Best Parameters of Light GBM Model](#7.15)   
    * [7.16. Build Light GBM Model with Best Parameters](#7.16)    
    * [7.17. Accuracy Score of Light GBM Model on Test set](#7.17)    
    * [7.18. Classification Report of Light GBM Model](#7.18)    
    * [7.19. ROC AUC - Light GBM Model](#7.19)
 
* <div style="border-radius:10px;
            border : black solid;
            background-color: #E8D9F3;
            font-size:100%;
            text-align: left">

<h2 style='; border:0; border-radius: 15px; font-weight: bold; color:black'><center> Explanation of the variables 📜 📖 📚
</center></h2>  
    
* ****fixed acidity:**** most acids involved with wine or fixed or nonvolatile (do not evaporate readily). Acidity is a characteristic determined by the total sum of acids that a sample contains. We can quantify the set of all of them in an undifferentiated way (total acidity) or in a grouped way (fixed acidity and volatile acidity). Fixed acidity corresponds to the set of low volatility organic acids such as malic, lactic, tartaric or citric acids and is inherent to the characteristics of the sample.    
* ****volatile acidity:**** the amount of acetic acid in wine, which at too high of levels can lead to an unpleasant, vinegar taste. Volatile acidity corresponds to the set of short chain organic acids that can be extracted from the sample by means of a distillation process: formic acid, acetic acid, propionic acid and butyric acid.    
* ****citric acid:**** found in small quantities, citric acid can add 'freshness' and flavor to wines. Citric acid is a colorless weak organic acid. It occurs naturally in citrus fruits. In biochemistry, it is an intermediate in the citric acid cycle, which occurs in the metabolism of all aerobic organisms.    
* ****residual sugar:**** the amount of sugar remaining after fermentation stops, it's rare to find wines with less than 1 gram/liter. Residual sugar refers to the sugars left unfermented in a finished wine. It is measured by grams of sugar per litre (g/l). The amount of residual sugar affects a wine's sweetness and, in the EU, the RS level is linked to specific labelling terms.    
* ****chlorides:**** the amount of salt in the wine. The higher extraction of chloride during red winemaking is due to the ions extracted from skins during fermentation. Therefore, red juice should have no more than 356mg/L chloride ions so that finished wine does not exceed the maximum legal level of 606mg/L chloride(356mg/L in red juice x 1.7 = 606).    
* ****free sulfur dioxide:**** the free form of SO2 exists in equilibrium between molecular SO2 (as a dissolved gas) and bisulfite ion. What is free sulphur dioxide in wine? The free sulfites are those available to react and thus exhibit both germicidal and antioxidant properties. The bound sulfites are those that have reacted (both reversibly and irreversibly) with other molecules within the wine medium. The sum of the free and bound sulfites defines the total sulfite concentration.    
* ****total sulfur dioxide:**** amount of free and bound forms of S02; in low concentrations, SO2 is mostly undetectable in wine, but at free SO2. Simply put, Total Sulfur Dioxide (TSO2) is the portion of SO2 that is free in the wine plus the portion that is bound to other chemicals in the wine such as aldehydes, pigments, or sugars.    
* ****density:**** the density of water is close to that of water depending on the percent alcohol and sugar content. How do you measure the density of wine? A hydrometer is an instrument used to measure liquid density. It is a sealed glass tube with a weighted bulb at one end, winemakers use this instrument to measure density of juice, fermenting wine and completed wine in relation to pure water. This ratio is called specific gravity (SG).    
* ****pH:**** describes how acidic or basic a wine is on a scale from 0 (very acidic) to 14 (very basic); most wines are between 3-4. What is a high pH in wine? Wines which have higher pH values (>3.65) have a series of potential challenges during vinification and aging. First, high pH wines have an increased chance of microbial spoilage. Traditionally, sulfur dioxide (often in the form of potassium metabisulfite) is used to keep wines stable during aging.    
* ****sulphates:**** a wine additive which can contribute to sulfur dioxide gas (S02) levels, wich acts as an antimicrobial. Wine sulfites are naturally occurring at low levels in all wines, and are one of the thousands of chemical by-products created during the fermentation process. However, sulfites are also added by the winemaker to preserve and protect the wine from bacteria and yeast-laden invasions. For some, sulfur allergies may be associated with headaches and stuffy sinuses after a glass or two of wine. It is a a wine additive which can contribute to sulfur dioxide gas (S02) levels, wich acts as an antimicrobial.    
* ****Alcohol:**** this is the percent alcohol content of the wine    
* ****quality:**** output variable (based on sensory data, score between 3 and 8).

![Stats](https://github.com/user-attachments/assets/4b51e7b1-3a7c-440e-a468-d4a059906594)

<div style="border-radius:10px;
            border : black solid;
            background-color: #E8D9F3;
            font-size:110%;
            text-align: left">

<h4 style='; border:0; border-radius: 15px; font-weight: bold; color:black'><center>Distribution</center></h4>  

Analyzing the graphs here, it turns out that the values of the variable <mark><b>'fixed_acidity'</b></mark> are relatively <mark><b>normally distributed (but a bit left skewed).</b></mark> But there are two peaks in the distributions of other <mark><b>'volatile_acidity'</b></mark> and <mark><b>'citric_acid'</b></mark> variables.
