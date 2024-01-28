import os 
import shutil
folder=r"D:\ricemodify\runRFDL\train\patchsize7\2022"
targetfolder=r"D:\ricemodify\runRFDL\train\patchsize7\plot"
n=0
for f in os.listdir(folder):
    file=os.path.join(folder,f)
    for sf in os.listdir(file):
        if sf.endswith('.png'):
           
            t=os.path.join(targetfolder,f'{n}.png')
            shutil.move(os.path.join(file,sf),t)
            n+=1
            