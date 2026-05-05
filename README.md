# Bakso Pajero ABSA Dataset

This project is a Natural Language Processing project for building an Aspect-Based Sentiment Analysis dataset from Google Places reviews of Bakso Pajero Colombo.

## 📌 Overview

The goal of this project is to collect, clean, annotate, and analyze Indonesian Google Places reviews using Aspect-Based Sentiment Analysis. The annotation focuses on four main aspects: Product, Price, Place, and Promotion. Reviews that do not discuss these aspects are labeled as Out of Topic.

The dataset is designed for multi-label ABSA, meaning one review can contain more than one aspect-sentiment label.

## 🏷️ ABSA Labels

The annotation schema includes the following labels:

- PRODUCT_POSITIVE
- PRODUCT_NEGATIVE
- PRODUCT_NEUTRAL
- PRICE_POSITIVE
- PRICE_NEGATIVE
- PRICE_NEUTRAL
- PLACE_POSITIVE
- PLACE_NEGATIVE
- PLACE_NEUTRAL
- PROMOTION_POSITIVE
- PROMOTION_NEGATIVE
- PROMOTION_NEUTRAL
- OUT_OF_TOPIC

## Streamlit

https://project-eat42rtlpmyfqyqqudw5dx.streamlit.app/

## 📁 Dataset

This project contains three main dataset files:

```txt
Kelp4_dataset_1.csv
Kelp4_dataset_2.csv
Kelp4_dataset_anotasi.jsonl
