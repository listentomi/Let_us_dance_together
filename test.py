#画图
import numpy as np
import cv2

import json

CANVAS_SIZE = (400,600,3)

def draw(frames):
    frames[:, :, 0]*=140
    frames[:, :, 1]*=140
    frames[:,:,0] += CANVAS_SIZE[0]//2
    frames[:,:,1] += CANVAS_SIZE[1]//3
    for i in range(len(frames)):
        cvs = np.ones(CANVAS_SIZE)*255
        color = (0,0,0)
        hlcolor = (255,0,0)
        dlcolor = (0,0,255)
        for j in range(len(frames[i])):
            points=frames[i][j]
            cv2.circle(cvs, (int(points[0]), int(points[1])), radius=4, thickness=-1, color=hlcolor)
            '''if(j==len(frames[i])-3):
                cv2.circle(cvs, (int(points[0]), int(points[1])), radius=4, thickness=-1, color=(0,255,0))
            else:
                cv2.circle(cvs, (int(points[0]), int(points[1])), radius=4, thickness=-1, color=hlcolor)'''

        frame = frames[i]
        # cv2.line(cvs, (int(frame[0][0]), int(frame[0][1])), (int(frame[1][0]), int(frame[1][1])), color, 2)
        # cv2.line(cvs, (int((frame[0][0]+frame[1][0])/2), int((frame[0][1]+frame[1][1])/2)), (int((frame[3][0]+frame[12][0])/2), int((frame[3][1]+frame[12][1])/2)), color, 2)
        # cv2.line(cvs, (int(frame[3][0]), int(frame[3][1])), (int((frame[3][0]+frame[12][0])/2), int((frame[3][1]+frame[12][1])/2)), color, 2)
        # cv2.line(cvs, (int(frame[3][0]), int(frame[3][1])), (int(frame[4][0]), int(frame[4][1])), color, 2)
        # cv2.line(cvs, (int(frame[4][0]), int(frame[4][1])), (int(frame[5][0]), int(frame[5][1])), color, 2)
        # cv2.line(cvs, (int(frame[5][0]), int(frame[5][1])), (int(frame[6][0]), int(frame[6][1])), color, 2)
        # cv2.line(cvs, (int(frame[12][0]), int(frame[12][1])), (int((frame[3][0]+frame[12][0])/2), int((frame[3][1]+frame[12][1])/2)), color, 2)
        # cv2.line(cvs, (int(frame[12][0]), int(frame[12][1])), (int(frame[13][0]), int(frame[13][1])), color, 2)
        # cv2.line(cvs, (int(frame[13][0]), int(frame[13][1])), (int(frame[14][0]), int(frame[14][1])), color, 2)
        # cv2.line(cvs, (int(frame[14][0]), int(frame[14][1])), (int(frame[15][0]), int(frame[15][1])), color, 2)
        # cv2.line(cvs, (int(frame[2][0]), int(frame[2][1])), (int((frame[3][0]+frame[12][0])/2), int((frame[3][1]+frame[12][1])/2)), color, 2)
        # cv2.line(cvs, (int(frame[2][0]), int(frame[2][1])), (int(frame[7][0]), int(frame[7][1])), color, 2)
        # cv2.line(cvs, (int(frame[7][0]), int(frame[7][1])), (int(frame[8][0]), int(frame[8][1])), color, 2)
        # cv2.line(cvs, (int(frame[8][0]), int(frame[8][1])), (int(frame[9][0]), int(frame[9][1])), color, 2)
        # cv2.line(cvs, (int(frame[9][0]), int(frame[9][1])), (int((frame[10][0]+frame[11][0])/2), int((frame[10][1]+frame[11][1])/2)), color, 2)
        # cv2.line(cvs, (int(frame[10][0]), int(frame[10]
        #                                       [1])), (int(frame[11][0]), int(frame[11][1])), color, 2)
        # cv2.line(cvs, (int(frame[2][0]), int(frame[2][1])), (int(frame[16][0]), int(frame[16][1])), color, 2)
        # cv2.line(cvs, (int(frame[16][0]), int(frame[16][1])), (int(frame[17][0]), int(frame[17][1])), color, 2)
        # cv2.line(cvs, (int(frame[17][0]), int(frame[17][1])), (int(frame[18][0]), int(frame[18][1])), color, 2)

        for j in range(20):
            cv2.putText(cvs,str(j),(int(frame[j][0]),int(frame[j][1])),cv2.FONT_HERSHEY_SIMPLEX,.4, (155, 0, 255), 1)
        cv2.imshow('canvas',np.flip(cvs,0))
        paper = np.flip(cvs, 0)
        paper = paper.astype(np.uint8)
        videoWriter.write(paper)
        cv2.waitKey(25)
    pass

if __name__ == '__main__':
    frames=np.empty([0,20,3])
    with open('./skeleton.out','r') as fin:
        count=0
        for line in fin.readlines():
            if(count%21==0):
                data=np.empty([0,3])
                line=line.split()
                line=list(map(float,line))
                for i in range(len(line)//3):
                    data=np.append(data,np.array([line[i*3],line[i*3+1],line[i*3+2]]).reshape([1,3]), axis=0)
                frames=np.append(frames,data.reshape([1,20,3]), axis=0)
            count+=1
frames=np.array(frames)
fps = 25
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
videoWriter = cv2.VideoWriter('visualization.avi', fourcc, fps, (600,400))

draw(frames)
#draw(np.array(data['skeletons']['skeletons']))