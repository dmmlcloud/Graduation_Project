import sys
import os

if __name__ == '__main__':
    with open("./colmap_test/model/images.txt", encoding='utf-8') as file:
        content = file.readlines()
        index = 0
        for line in content:
            if index > 3:
                if(index % 2 == 1):
                    data = line.split(' ')
                    for i in range(int(len(data) / 3)):
                        if(float(data[i * 3]) >= 950):
                            print(data[i * 3 + 0], " ", data[i * 3 + 1], " ", data[i * 3 +2])
            index += 1
