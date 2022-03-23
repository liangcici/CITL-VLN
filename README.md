# Contrastive Instruction-Trajectory Learning for Vision-Language Navigation

This is the code for the [CITL](https://arxiv.org/abs/2112.04138) paper.

## Installation
1. Python requirements: Need python3.6 or higher and install dependencies:
    ```
    pip install -r requirements.txt
    ```
2. Install [Matterport3D simulator](https://github.com/peteanderson80/Matterport3DSimulator). Notice that this code uses the [old version (v0.1)](https://github.com/peteanderson80/Matterport3DSimulator/tree/v0.1) of the simulator.
3. Download resources in the interactive Python interpreter:
    ```
    import stanfordnlp
    stanfordnlp.download('en')   # This downloads the English models for the neural pipeline
    import nltk
    nltk.download('averaged_perceptron_tagger')
    nltk.download('omw-1.4')
    ```

## Data Preparation
1. Follow [Recurrent VLN-BERT](https://github.com/YicongHong/Recurrent-VLN-BERT#recurrent-vln-bert) to download data.
2. Generate sub-instructions, positive instructions and sub-optimal trajectories:
    ```
   sh scripts/generate_augment_data.sh
   ```
   
## Navigator
- Training
    ```
    bash scripts/train_agent.bash
    ```
    It takes several days (~4 days) to train on a single GPU.
- Testing
    ```
    bash scripts/test_agent.bash
    ```


## Acknowledgement
The implementation relies on resources from [Recurrent VLN-BERT](https://github.com/YicongHong/Recurrent-VLN-BERT#recurrent-vln-bert) and [Fine-Grained R2R](https://github.com/YicongHong/Fine-Grained-R2R). We thank the original authors for their open-sourcing.

## Reference
If you find this code useful, please consider citing.
```
@article{liang2021contrastive,
  title={Contrastive Instruction-Trajectory Learning for Vision-Language Navigation},
  author={Liang, Xiwen and Zhu, Fengda and Zhu, Yi and Lin, Bingqian and Wang, Bing and Liang, Xiaodan},
  journal={arXiv preprint arXiv:2112.04138},
  year={2021}
}
```
