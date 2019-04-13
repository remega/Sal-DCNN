Sal-DCNN
==========
The model of [**"Image Saliency Prediction in Transformed Domain: A Deep Complex Neural Network Method"**](https://www.dropbox.com/s/78tbqy82u6ne3m5/AAAI-JiangL.3574.pdf?dl=0), which has also been published at AAAI2019.

## Abstract
The transformed domain fearures of images show effectiveness in distinguishing salient and non-salient regions. In this paper, we propose a novel deep complex neural network, named Sal-DCNN, to predict image saliency by learning features in both pixel and transformed domains. Before proposing Sal-DCNN, we analyze the saliency cues encoded in discrete Fourier transform (DFT) domain. Consequently, we have the following findings: 1) the phase spectrum encodes most saliency cues; 2) a certain pattern of the amplitude spectrum is important for saliency prediction; 3) the transformed domain spectrum is robust to noise and down-sampling for saliency prediction. According to these findings, we develop the structure of Sal-DCNN, including two main stages: the complex dense encoder and three-stream multi-domain decoder. Given the new Sal-DCNN structure, the saliency maps can be predicted under the supervision of ground-truth fixation maps in both pixel and transformed domains. Finally, the experimental results show that our Sal-DCNN method outperforms other 8 state-of-the-art methods for image saliency prediction on 3 databases.

## Publication
If you are interested in this method please cite:  
```
@article{jiang2019saldcnn,
  title={Image Saliency Prediction in Transformed Domain: A Deep Complex Neural Network Method},
  author={Lai Jiang, Zhe Wang, Mai Xu, Zulin Wang},
  booktitle = {AAAI Conference on Artificial Intelligence (AAAI)},
  month = {February},
  year = {2019} 
}
```

## Models
The pre-trained model can be found in [dropbox](https://www.dropbox.com/sh/t5brryoagx4l7ye/AACpa5lUkqqjPCsChCUuNwfya?dl=0).
For running the demo, please downloard the model to the directory of **./model/**.

![Sal-DCNN](/fig/SalDCNN.png "Sal-DCNN")


## Usage
This model is implemented by **tensorflow-gpu** 1.10.0, and the detail of our computational environment is listed in **'env.txt'**. 
Run **'TestSALDCNN.py'** to get the saliency prediction results over the images put in **./img/**.

## Results
The results are output to  **./result/**.
Some results of our model and ground-truth.
![Results](/fig/res.png "Results")

## Contact
If any question, please contact jianglai.china@buaa.edu.cn （or jianglai.china@gmail.com）, or use public issues section of this repository.

## License
This code is distributed under MIT LICENSE.
