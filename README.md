# LoD3Framework-A Framework for Fully Automated Reconstruction of Semantic Building Model at Urban-Scale Using Textured LoD2 Data
The proposed reconstruction framework is divided into three parts. The first part is the parsing of the LoD2 model, which you can find the detailed process in readgml1.py. The second part is the reconstruction of semantic facade models, which includes initpara_batch3_stage, inverse_facade_stage, and amendwindow22_stage_PATCH, which need to work together to ensure the correct operation of this step. The third part is Exporting: LoD3 Model Generation, which is completed by create1. At the same time, the control commands for the three processes are also recorded in Main_auto.py.
# Configuration environment:
In this project, the environment setup only requires attention to the requirements of Mask RCNN. The rest of the parts have been validated in Python 3.6/3.7/3.8/3.9, which are fully functional and require common Python dependency libraries for computation.
Mask RCNN Weight can be download at: https://drive.google.com/drive/folders/1PgR3OMujgK-_J67Ml7gYVbQ0YSdFR0Gd?usp=sharing
Textured LoD2 data can be download at: https://drive.google.com/drive/folders/1MmJRyNzRVZ1yVqRkT4hVGcggT2aJaAWf?usp=sharing
# Excuting step
We have integrated the three steps mentioned into a single Main_auto.py file. However, since we are running the instance segmentation task on a remote server, you will see "# --waiting for the result of Mask-RCNN----" in Main_auto. At this point, you need to follow the instructions in facade_maskrcnn.rar to run the instance segmentation program to generate a file that records semantic entity information, such as "ces2". Another thing to note is that we store the extracted relationships in the img_model.pkl file, which you can read and view the relationship between facade and image.

# Other notes:
We have validated this framework in a model based on photogrammetric mesh. If you want more information, please don't hesitate to contact me:Email:wangyuefeng2017@whu.edu.cn; wangyuefeng2021@glut.edu.cn
This work has been published in ISPRS J P&RS. You can cited by "Wang, Y., Jiao, W., Fan, H., & Zhou, G. (2024). A framework for fully automated reconstruction of semantic building model at urban-scale using textured LoD2 data. ISPRS Journal of Photogrammetry and Remote Sensing, 216, 90-108".
