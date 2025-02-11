# CatGWR: Context-Attention Geographically Weighted Regression

<div align="center">
<img alt="A flat-design illustration in landscape orientation featuring a sleek black cat in a walking posture as a silhouette, occupying a larger portion of the scene. The background includes minimalist and abstract elements such as a globe or map representing geography, small graphs with dots and lines symbolizing geographically weighted regression, interconnected nodes for neural networks, and arrows highlighting specific nodes to represent attention mechanisms. The design is modern and clean, using soft, muted tones for the background to maintain simplicity. No text is included in the background." height="366" src="./figures/CatGWR_mascot.webp"/>  
  
generated with DALL‧E-3
</div>
  
## Description

This method is an extension of GWR (Geographically Weighted Regression) model. The related article is published on _International Journal of Geographical Information Science_

> Wu, S., Ding, J., Wang, R., Wang, Y., Yin, Z., Huang, B., & Du, Z. (2025). Using an attention-based architecture to incorporate context similarity into spatial non-stationarity estimation. _International Journal of Geographical Information Science_, 1–24. https://doi.org/10.1080/13658816.2025.2456556


Context similarity, as a complement to spatial proximity, is introduced in geographically weighted regressions. An attention-based framework is developed to calculate the context similarity between locations.


## How to use

- Run the `run-cv.py` file with python3. Hyperparameters and options can also be set in this file.
- Modify the `models.py` file if you want to change the network structure. 

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

### Attribution
This implementation is based on [PyTorchGAT](https://github.com/gordicaleksa/pytorch-GAT) by Gordić, Aleksa, 2020, licensed under the MIT License. 

The original license is included in the `LICENSE` file.

