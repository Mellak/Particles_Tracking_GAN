# Fast-Track F-18 Positron Paths Simulations: An ISBI Paper Overview

## Introduction

Our project, inspired by the findings presented in our ISBI paper titled "Fast-Track of F-18 Positron paths simulations," introduces an innovative deep learning-based method for simulating the trajectories of F-18 positrons. Utilizing a Generative Adversarial Network (GAN), we offer a rapid and accurate alternative to traditional Monte Carlo (MC) simulations, providing critical insights into positron interactions across different materials.

## Key Achievements

- **Precise Simulation of Positron Paths**: Through the adept use of GANs, our model accurately replicates the Point Spread Function (PSF) distributions of positrons as they traverse through water, bone, and lung materials, closely matching the results obtained from GATE simulations.
- **Innovative Comparative Analysis**: Our comprehensive analysis, including a detailed examination of mean and maximum radii of positron paths, reveals a less than 10% difference in maximum path lengths compared to traditional methods, underscoring the model's precision.
- **Exceptional Efficiency**: The proposed model significantly outpaces conventional MC simulations in terms of speed, requiring only 6 seconds to generate a 20000 of events, compared to approximately 45 seconds with GATE, thereby enhancing computational efficiency.

## Features

- **Accurate Path Simulations**: Generates paths with mean lengths and energy distributions closely approximating those produced by traditional GATE simulations.
- **Enhanced Computational Speed**: Achieves a significant reduction in simulation time, facilitating the generation of large datasets for in-depth analysis.
- **Versatile Material Analysis**: Capable of simulating positron paths in various materials, including water, bone, and lung, with plans to extend to other radionuclides and volumes.

## Discussion and Future Directions

The proposed method not only demonstrates a high degree of accuracy in simulating positron paths but also introduces a paradigm shift in particle tracking, leveraging the power of GANs to model particle trajectories as sequences. Key to our approach is the conditioning of the GAN model on initial energy and interaction counts, allowing for the simulation of varied interaction paths. The inclusion of a cosine loss term further refines the model's ability to approximate initial particle directions accurately.

Future work will aim to extend the model's capabilities to heterogeneous materials and other radionuclides, broadening the scope of applications for this promising methodology.

## Contributions and Acknowledgments

This work represents a collaborative effort within the biomedical imaging and simulation community. We extend our gratitude to all contributors, especially those who provided data and feedback. For contributions, please refer to the Contribution section for guidelines.

## License

This project is licensed under the MIT License.