## Overview
This repository is a quick experimentation platform for new AI Methodologies.

It provides validated PoC for AI Modelling in:

- **Feature-Feature Attention Maps MultivariateTime Series Analysis** (iTransformer-backbone) — By tokenizing sensor features rather than time steps (i.e. feature-as-a-token) , this approach learns which signals have the most importance for a predictive (or a non-predictive) task, enabling, for example, anomaly attribution to specific sensor signals, by identifying their centrality in the attention graph. The key point is to understand which signals most sell and buy information from and to the others, in a multivariate time-series dynamic system, i.e. where features are correlated to each other (e.g., vehicle dynamics). Think of it as a way to answer not just "something is wrong" but "these signals, at this moment, are why."   

- **Multi-Head Autocorrelation Attention Maps** (ViTime-backbone) — In this example, you can see how long-term dependencies are captured by Attention Maps outperforming traditional ML algorithms and maintaining model explainability. The key point is exploiting Attention Maps to interpret temporal auto-correlation of signals in long-term relationships. To prove this, an ablation example is reported that demonstrates harmonic pattern identification by self-attention heads.
  
- **Attention Graph Neural Network** development for Energy Price Forecasting in RT Market.

This repo focuses on the reusable foundations rather than proprietary implementations, it builds layer of AI models to be cross used among projects.
