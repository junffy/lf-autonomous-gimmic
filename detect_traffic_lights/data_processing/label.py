import sys
import os
import commands
import subprocess

data_dir = './label/'

def cmd(cmd):
	return commands.getoutput(cmd)

dirs = cmd("ls "+sys.argv[1])
backup_dir = os.path.dirname(os.path.abspath(__file__)) + "/test_buckup_dir"
labels = dirs.splitlines()

print(data_dir)
print(backup_dir)


if os.path.exists(data_dir):
    cmd("rm  -rf "+data_dir)

if os.path.exists(backup_dir):
    cmd("rm  -rf "+backup_dir)


os.makedirs(data_dir+"/images")
os.makedirs(backup_dir)

pwd = cmd('pwd')
imageDir = data_dir+"/images"


train = open(data_dir + '/train.txt','w')
train_lstm = open(data_dir + '/train_lstm.tsv','w')

test = open(data_dir + '/test.txt','w')

labelsTxt = open(data_dir + '/labels.txt','w')
labelsTxt_backup = open(backup_dir + '/labels.txt','w')

classNo=0
cnt = 0

for label in labels:
    workdir = pwd+"/"+sys.argv[1]+"/"+label
    imageFiles = cmd("ls "+workdir+"/*.jpg")
    images = imageFiles.splitlines()
    labelsTxt.write(label+"\n")
    labelsTxt_backup.write(label+"\n")
    startCnt=cnt
    length = len(images)
    print("----file num----")
    print(length)
    print("--------")

    for image in images:
        imagepath = imageDir+"/image%07d" %cnt +".jpg"
        cmd("cp "+image+" "+imagepath)

        print(imagepath)

        if cnt-startCnt < length*0.75:
            train.write(imagepath+" %d\n" % classNo)
            train_lstm.write(imagepath+"\t%s\n" % label)
            print("     ----train data----")
        else:
            test.write(imagepath+" %d\n" % classNo)
            print("     ----test data----")
        cnt += 1
    classNo += 1