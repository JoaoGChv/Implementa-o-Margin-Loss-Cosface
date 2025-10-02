<div align="center">

# Extração de Embeddings Faciais com Deep Metric Learning

### Um pipeline robusto para treinamento e avaliação de modelos de reconhecimento facial usando CosFace e Transfer Learning com TensorFlow/Keras.

</div>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/TensorFlow-2.15-orange.svg" alt="TensorFlow">
  <img src="https://img.shields.io/badge/PyTorch-2.1-red.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/Licença-MIT-green.svg" alt="License">
</p>

---

## 📖 Visão Geral

Este projeto implementa um pipeline completo para treinamento e avaliação de modelos de reconhecimento facial, com foco na extração de **embeddings faciais** altamente discriminativos. Utilizando a função de perda **CosFace (Large Margin Cosine Loss)**, o objetivo é treinar uma rede neural que mapeia imagens de faces para um espaço vetorial, onde a semelhança semântica (mesma identidade) se traduz diretamente em proximidade geométrica.

O trabalho partiu de uma implementação base do [keras-arcface](https://github.com/4uiiurz1/keras-arcface) e foi extensivamente refatorado e aprimorado, evoluindo de uma prova de conceito no dataset MNIST para um pipeline robusto capaz de treinar em datasets de larga escala como o **VGGFace2** e validar no benchmark padrão **LFW**.

## ✨ Principais Melhorias e Funcionalidades

Este projeto representa uma evolução significativa da base original, focando em robustez, escalabilidade e práticas de engenharia de software para Machine Learning.

-   **🚀 Transição para Datasets de Larga Escala:**
    -   **De MNIST para Faces:** O pipeline foi completamente adaptado para treinar no **VGGFace2** (~3M de imagens, ~9k identidades).
    -   **Validação Padrão de Mercado:** A avaliação de performance é realizada no **LFW (Labeled Faces in the Wild)**, permitindo a comparação com resultados acadêmicos.

-   **🧠 Arquitetura e Treinamento:**
    -   **Backbone Profissional:** Substituição da arquitetura VGG8 por uma **ResNet50 pré-treinada** na ImageNet.
    -   **Fine-Tuning em Duas Fases:** Implementação de uma estratégia de treinamento estável que primeiro treina a "cabeça" do modelo e depois realiza o *fine-tuning* do backbone, garantindo uma convergência eficaz.

-   **📊 Análise e Avaliação usada:**
    -   **Métricas Completas:** O treinamento monitora **Acurácia**, **Precisão** e **Recall**.
    -   **Extração de Embeddings:** O pipeline de inferência salva os embeddings gerados em um arquivo **JSON** para análise detalhada.
