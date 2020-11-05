
<p align="center">
    <a href="/docs/readme_en.md">English </a>
    ·
    <a href="/README.md">Português Brasileiro</a>

</p>

# GERAÇÃO AUTOMÁTICA DE TEXTOS DESCRITIVOS DE UM VÍDEO COM USO DE APRENDIZAGEM DE MÁQUINA

Codigo desenvolvido para Tese de Conclusao de Curso.

## Pre-requisitos

```
•python 3.6 (sudo apt-get install python3.6)
•tensorflow 2.2.0 (pip install tensorflow==2.2.0)
•numpy 1.18.5 (pip install numpy)
•keras 2.4.3 (pip install keras)
•opencv 4.3.0 (pip install opencv-python)
•matplotlib 3.2.2 (pip install matplotlib)
•pillow 7.1.2 (pip install Pillow)
•nltk 3.5.0 (pip install Pillow)
•CMake 3.12 (pip install cmake)
•pickle5 0.0.11 (pip install picke5)

```
- [YOLOV4](https://github.com/AlexeyAB/darknet)
- [Dados do MSCOCO](https://cocodataset.org/#download)

## Como usar
### Preparar os dados de texto
No `/GenerateCaptions`

Para traduzir as anotacoes de ingles para portugues:
`python3 JsonToRawTxt.py`

Colocar o documento de output para traduzir

Para criar um novo arquivo json com anotacoes traduzidas:
`python3 textToTranslatedJson.py`

Json resultante vai ser usado para treinar a rede

### Preparar os dados de imagem
Se quiser pegar keyframes de um video, no `/` executar:
`python3 -m KeyframeExtraction`

 Caso contrario executar:
`python3 createTrainImageList.py`

Pegar o arquivo resultante `train.txt` e executar YOLOv4:
`./darknet detector test cfg/coco.data cfg/yolov4.cfg yolov4.weights -dont_show -ext_output < data/train.txt > result.txt`

 A partir do result.txt do Yolov4 e as imagens de treino, criar a base de treino para esse programa:
`python3 -m FeatureGenerator`

### Treino da rede
A partir dos dados gerados, treinar a rede:
`python3 -m ImageCaption_train`

Testar os resultados:
`python3 -m ImageCaption_validation`

## References
<!-- https://www.pugetsystems.com/labs/hpc/ 
How-To-Install-CUDA-10-together-with-9-2-on-Ubuntu-18-04-with-support-for-NVIDIA-20XX-Turing-GPUs-1236/
howtogeek.com/442101/how-to-move-your-linux-home-directory-to-another-hard-drive/-->


https://www.tensorflow.org/tutorials/text/image_captioning

https://github.com/iShoto/testpy/tree/master/samples/image_clustering


## License



## Acknowledgments
