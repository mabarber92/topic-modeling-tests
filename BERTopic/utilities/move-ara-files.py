import os
from tqdm import tqdm
import shutil

def copySpecFiles(dir, newDir, ext="-ara", newExt=".csv"):
    for root, dirs, files in os.walk(dir, topdown=False):
            for name in tqdm(files):
                if ext in name:
                    print(name)
                    newName = name.split("-")[0] + newExt
                    newPath = newDir + "/" + newName
                    oldPath = os.path.join(root, name)
                    shutil.copyfile(oldPath, newPath)


if __name__ == '__main__':
    oldDir = "C:/Users/mathe/Documents/Github-repos/topic-modeling-tests/wholeCorpusOld/summaries"
    newDir = "C:/Users/mathe/Documents/Github-repos/topic-modeling-tests/wholeCorpus/summaries"
    copySpecFiles(oldDir, newDir)
    oldDir = "C:/Users/mathe/Documents/Github-repos/topic-modeling-tests/wholeCorpusOld/topicsByText"
    newDir = "C:/Users/mathe/Documents/Github-repos/topic-modeling-tests/wholeCorpus/topicsByText"
    copySpecFiles(oldDir, newDir)
