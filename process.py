import cv2 as cv
import numpy as np
import pandas as pd
import zipfile as zf
import json as js
import random as rd
import copy
import io
import os
from cv2 import cv2 

## JSON
class ProcessJSON:
    # init
    def __init__(self):
        self.json = None
        self.jsonMerge = None
        self.jsonSplit = []

    # 객체 초기화
    def valueClear(self):
        self.json = None
        self.jsonMerge = None
        self.jsonSplit = []

    # get
    def getJson(self):
        return self.json

    def getJsonSplit(self):
        return self.jsonSplit

    def getJsonMerge(self):
        return self.jsonMerge

    # json 불러오기
    def jsonFileLoad(self, jsonFilePath, jsonFileName):
        try:
            jsonFilePathName = jsonFilePath + '/' + jsonFileName
            processJSON = open(jsonFilePathName, 'r')
            self.json = js.load(processJSON)
        except Exception as ex:
            return False
        return True

    # json 파일 분할하기
    def jsonFileSplit(self, jsonFilePath, jsonFileName):
        try:
            jsonFilePathName = jsonFilePath + '/' + jsonFileName
            processJSON = open(jsonFilePathName, 'r')
            # 원본 json
            self.jsonMerge = js.load(processJSON)
            self.jsonSplit = []

            # 1 대 목록별 분할
            images = self.jsonMerge['images']
            categories = self.jsonMerge['categories']
            annotations = self.jsonMerge['annotations']

            # 2 이미지를 기준으로 카테고리 및 어노테이션 추가
            for image in images:
                # 2.1 dataset 초기화
                oneJson = {'images': [], 'categories': [], 'annotations': []}
                # 2.2 이미지추가
                oneJson['images'].append(image)

                # 2.3 categories & annotations
                for category in categories:
                    for annotation in annotations:
                        # 2.4 어노테이션을 기준으로 카테고리와 이미지의 id를 비교하여 둘다 동일하다면
                        if image['id'] == annotation['image_id'] and category['id'] == annotation['category_id']:
                            oneJson['categories'].append(category)
                            oneJson['annotations'].append(annotation)

                self.jsonSplit.append(oneJson)
        except Exception as ex:
            return False
        return True

    # json 분할 데이터 저장하기
    def jsonFileSplitSave(self, jsonFilePath):
        for jsonSplit in self.jsonSplit:
            try:
                # 1. 파일명 재명
                jsonFileName = jsonSplit['images'][0]['file_name'] + '.json'
                # 2. 경로 및 설치 지정
                jsonFilePathName = jsonFilePath + '/' + jsonFileName
                # 3. 파일 쓰기 준비
                processJSON = open(jsonFilePathName, 'w')
                # 4. json 파일 저장
                js.dump(jsonSplit, processJSON)
                # 5. 저장 파일 닫기
                processJSON.close()
            except Exception as ex:
                pass

    # json 파일 합치기
    def jsonFileMerge(self, jsonFilePath):
        try:
            # 1. 파일 리스트 불러오기
            fileList = os.listdir(jsonFilePath)

            # 2. json 뭉칠 파일 설정
            self.jsonMerge = {'images': [], 'categories': [], 'annotations': []}
            for jsonFileName in fileList:
                jsonFilePathName = jsonFilePath + '/' + jsonFileName
                processJSON = open(jsonFilePathName, 'r')
                oneJson = js.load(processJSON)

                self.jsonMerge['images'].append(oneJson['images'][0])
                for category in oneJson['categories']:
                    check = True
                    # 3. 카테고리 중복확인
                    for mergeCategory in self.jsonMerge['categories']:
                        if category == mergeCategory:
                            check = False
                            break

                    if check == True:
                        self.jsonMerge['categories'].append(category)

                for annotation in oneJson['annotations']:
                    self.jsonMerge['annotations'].append(annotation)
        except:
            return False
        return True

    # json 파일 합치고 자동으로 ID 부여하기
    def jsonFileMergeAndRedefineIndex(self, jsonFilePath, options):
        # 0. 전처리 시작점 분할하기
        datasetID = int(options['datasetID'])
        imagesID = int(options['imagesID'])
        categoriesID = int(options['categoriesID'])
        annotationsID = int(options['annotationsID'])
        filePath = options['filePath']

        # 1. 파일 리스트 불러오기
        fileList = os.listdir(jsonFilePath)
        jsonFileList = [file for file in fileList if file.endswith(".json")]

        self.jsonMerge = {'images': [], 'categories': [], 'annotations': []}
        for jsonFileName in jsonFileList:
            jsonFilePathName = jsonFilePath + '/' + jsonFileName
            processJSON = open(jsonFilePathName, 'r')
            oneJson = js.load(processJSON)

            oneJson['images'][0]['id'] = imagesID
            oneJson['images'][0]['dataset_id'] = datasetID
            oneJson['images'][0]['path'] = filePath + '/' + oneJson['images'][0]['file_name']
            self.jsonMerge['images'].append(oneJson['images'][0])

            for category in oneJson['categories']:
                check = True
                for mergeCategory in self.jsonMerge['categories']:
                    if category == mergeCategory:
                        check = False
                        break

                if check == True:
                    self.jsonMerge['categories'].append(category)

            for annotation in oneJson['annotations']:
                annotation['id'] = annotationsID
                annotation['image_id'] = imagesID
                self.jsonMerge['annotations'].append(annotation)
                annotationsID += 1

            imagesID += 1

    # 합친 json 파일 저장
    def jsonFileMergeSave(self, jsonFilePath, jsonFileName):
        try:
            jsonFilePathName = jsonFilePath + '/' + jsonFileName
            # 4. json 파일 저장
            processJSON = open(jsonFilePathName, 'w')
            # 4. json 파일 저장
            js.dump(self.jsonMerge, processJSON)
            # 5. 저장 파일 닫기
            processJSON.close()
        except Exception as ex:
            pass


class IMAGECONVERSION:
    def __init__(self):

        #index(0) : index(0) : seg값, id, 카테고리 넘버
        self.segmentations = {}

        # self.categoryname = {}

        #id : fiil_name
        self.ImagesDict = {}

        #id : segmentation, category
        self.key_segment = {}

        #변화율
        self.PercentValues = [-0.1, -0.09, -0.08, -0.07, -0.06, -0.05, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]

        #이미지가 존재하는 파일 경로
        # self.ImageFilePath = 'C:/Users/Trip1/Desktop/Python/TEST/'         

        #최종 이미지를 저장 할 경로
        # self.ImageSavePath = 'C:/Users/Trip1/Desktop/python/SaveTEST/'  

        #Color값
        self.ColorValues = 255 

        #histogram에 쓰일 list[ id, filename, image ]
        self.IdFileNameImageResult = [] 

        #json에 쓰일 list [ id, imagename ]
        self.jsonlist = []         

        #json에 쓰일 dict { id : 최종이미지 }
        self.key_dick = {}


    #딕셔너리 세팅
    def datasetting(self, json_data):
        index = 0

        #annotations 사용하기
        for annotations in json_data['annotations']:

            #딕셔너리 만들기, index(0) : seg값, id, 카테고리 넘버
            self.segmentations.setdefault(index, (annotations['segmentation'], annotations['image_id'], annotations['category_id']))
            index += 1

        #images 사용하기 
        for images in json_data['images']:

            #딕셔너리 만들기, id : fiil_name
            self.ImagesDict.setdefault(images['id'], images['file_name'])
            # print('ok')
    

    #최종 이미지명 
    def ImageNameResult(self,img_path, percent):
        
        #파일이름과 .jpg나누기
        path_list = img_path.split('.')
        
        #파일명 뒤에 _X% 값으로 이름 바꾸기
        x = '_{}%.'.format(int(percent*100))
        
        #파일명 합치기
        FinalName = path_list[0] + x + path_list[1]
        return FinalName
        

    #이미지를 불러오기 위한 이미지 경로/이미지명
    def LoadImagePath(self,ImageName, ImageFilePath):
        
        #파일경로, 이미지명
        ImageFileNamePath = ImageFilePath + ImageName
        return ImageFileNamePath


    #이미지 mask
    def mask_image(self,img,contours):

        #이미지 복사
        img2 = img.copy()

        #형식에 맞춘 3차원 행렬
        contours = contours.reshape((contours.shape[0],1,contours.shape[1]))

        #mask 사이즈
        mask = np.zeros(img.shape[:-1],np.uint8)

        #컨투어 좌표 내부(가방)를 채움
        cv2.fillPoly(mask, [contours], 255, cv2.LINE_AA)

        #가방부분(255), 배경(0) : 배경을 다시 흰색으로 변환 
        img2[mask ==0] = (255,255,255)

        # cv2.imwrite("masktest.jpg", mask)
        return mask
        

    #mask에 필요한 segmentation 반환
    def return_contours(self,segmt):

        #segmt값 불러오기 
        segmt = np.array(segmt)

        #segmt의 shape (0, 1, 2, 3  ...)즉, 1개 이외의 값이 들어있는 경우 
        if segmt.shape[0] != 1:
            segmt_len_list = []
            
            #segmt의 좌표의 개수를 list로 받는다
            for seg in segmt:
                segmt_len_list.append(len(seg))

            #가장 많은 좌표수만 가져오기
            segmet_index = segmt_len_list.index(max(segmt_len_list))
            segmt = segmt[segmet_index]
            segmt = np.array(segmt)

        #형식 맞춰주기    
        segmt = segmt.reshape(segmt.size)
        
        #짝수 x좌표, 홀수 y좌표
        segmt_x = segmt[::2]
        segmt_y = segmt[1::2]

        #컨투어 2차원 배열 만들기
        contours = np.zeros((len(segmt_x),2))

        #fillpoly()형식의 맞게 [x좌표 y좌표], 정수로 바꿈
        contours[:,0] = segmt_x.astype(np.int32); contours[:,1] = segmt_y.astype(np.int32)
        contours = contours.astype(np.int32)
        return contours
   

    #percent에 따른 이미지 변환하기 
    def ImageConvert(self,ImageFilePath, ImageSavePath):
        for id in self.ImagesDict:

            #이미지명 가져오기
            ImageName = self.ImagesDict[id]

            #파일경로/이미지명.jpg 가져오기
            ImagePath = self.LoadImagePath(ImageName, ImageFilePath)

            #이미지 불러오기
            ImageOriginal = cv2.imread(ImagePath, cv2.IMREAD_COLOR)  
        
            #각각의 퍼센트값의 따른 이미지 변환
            for a in range(0, len(self.PercentValues)):

                #퍼센트가 양수일 때 
                if self.PercentValues[a] > 0:

                    #이미지 원본 + 변환된 이미지
                    ImageResult = ImageOriginal + ImageOriginal*self.PercentValues[a]

                    #이미지의 색 정보가 255이상이면 흰색(255,255,255)으로 한다
                    ImageResult = np.where(ImageResult > (self.ColorValues,self.ColorValues,self.ColorValues), (self.ColorValues,self.ColorValues,self.ColorValues), ImageResult)

                #퍼센트가 음수일 때
                elif self.PercentValues[a] < 0:

                    #이미지 원본 + 변환된 이미지
                    ImageResult = ImageOriginal + ImageOriginal*self.PercentValues[a]

                    #배경도 어두어 지는 것을 막기위하여, 변화된 비율을 전부 흰색(255, 255, 255)
                    #수식 : 255-(255*퍼센트)
                    ImageResult = np.where(ImageResult >= (self.ColorValues-(self.ColorValues*abs(self.PercentValues[a])),self.ColorValues-(self.ColorValues*abs(self.PercentValues[a]))\
                        ,self.ColorValues-(self.ColorValues*abs(self.PercentValues[a]))), (self.ColorValues,self.ColorValues,self.ColorValues), ImageResult)

                #이미지 데이터 형식     
                ImageResult = ImageResult.astype(np.uint8)

                #최종 이미지명(파일경로/이미지명_X%.jpg)
                FileSavePathName = self.ImageNameResult(ImageSavePath + ImageName, self.PercentValues[a])

                #이미지명_X%.jpg 
                ImagePercentName = self.ImageNameResult(ImageName, self.PercentValues[a])

                #이미지 보정에 쓰일 : 아이디, 최종 파일명, 최종 이미지
                self.IdFileNameImageResult.append((id, FileSavePathName, ImageResult))

                #JSON저장에 쓰일 : 사용아이디, 이미지명  
                self.jsonlist.append((id, ImagePercentName))


    #딕셔너리 만들기, image_id : seg,category_id
    def IdDict(self):

        #key값만 모두 불러오기
        for key in self.ImagesDict.keys():

            #키값을 받고 value를 None으로 만든다
            self.key_segment.setdefault(key)

            #키값일 때 [] 만듬
            self.key_segment[key] =[]

        #key_segmnet의 값을 복사
        self.key_dick = copy.deepcopy(self.key_segment)     
        
        #image_id일 때 : seg, 카테고리 아이디 
        for seg,image_id,category_id in self.segmentations.values():
            self.key_segment[image_id].append((seg,category_id))   
                       

    #이미지 보정           
    def histogram(self):

        #아이디, 이미지 패스/이미지명.jpg, 변환된 이미지 불러오기
        for id, FileSavePathName, ImageResult in self.IdFileNameImageResult:

            #key_segment id에 따른 값 불러오기 
            key_Seglist = self.key_segment[id] 


            #이미지 불러오기, HSV 형태로 변환 때 
            

            #BGR -> HSV로 이미지 컬러 변경
            img_HSV = cv2.cvtColor(ImageResult,cv2.COLOR_BGR2HSV)

            #21, 22 카테고리빼기 (앰블럼)
            for seg_catgoryid in key_Seglist:
                if seg_catgoryid[1] != 21 and seg_catgoryid[1] != 22:

                    #seg_catgoryi[0](seg값)
                    contours = self.return_contours(seg_catgoryid[0])

                    #return_contours()에서 컨투어 받음 
                    mask = self.mask_image(ImageResult,contours)

                    #mask 씌우기 
                    img_HSV[mask == 0] = 0

                    #H(색상), S(채도), V(명도) split
                    H, S, V = cv2.split(img_HSV)

                    #배경부분을 제외한 S의 히스토그램 평활화
                    equalizeS = cv2.equalizeHist(S[S!=0])              

                    #배경부분을 제외한 V의 히스토그램 평활화 
                    #equalizeV = cv2.equalizeHist(V[V!=0])

                    #히스토그램 = None 일 때
                    if equalizeS is None:  
                        continue

                    #S의 shape맞추기
                    S[S!=0] = equalizeS.reshape(equalizeS.shape[0])

                    #V의 shape맞추기
                    #V[V!=0] = equalizeV.reshape(equalizeV.shape[0])

                    #H, S, V 합치기
                    img_ycbcr2 = cv2.merge([H, S, V])

                    #BGR로 변경                     
                    HistImageResult = cv2.cvtColor(img_ycbcr2, cv2.COLOR_HSV2BGR)

                    #배경 흰색
                    HistImageResult[mask == 0] =255,255,255

                    #이미지 저장
                    cv2.imwrite(FileSavePathName, HistImageResult)

                    
                 

            #이미지 불러오기, YCrCb 형태로 변환 때 
                    '''
            #BGR -> YCrCb로 이미지 컬러 변경
            img_ycbcr = cv2.cvtColor(ImageResult,cv2.COLOR_BGR2YCrCb)

            #21, 22 카테고리빼기 (앰블럼)
            for seg_catgoryid in key_Seglist:
                if seg_catgoryid[1] != 21 and seg_catgoryid[1] != 22:

                    #seg_catgoryi[0](seg값)
                    contours = self.return_contours(seg_catgoryid[0])

                    #return_contours()에서 컨투어 받음 
                    mask = self.mask_image(ImageResult,contours)

                    #mask 씌우기 
                    img_ycbcr[mask == 0] = 0

                    #Y(휘도), Cb,Cr(색차) split
                    Y, Cb, Cr  = cv2.split(img_ycbcr)                   

                    #배경부분을 제외한 Y의 히스토그램 평활화
                    equalizeY = cv2.equalizeHist(Y[Y!= 0])

                    #배경부분을 제외한 Cb의 히스토그램 평활화
                    equalizeCb = cv2.equalizeHist(Cb[Cb!=0])

                    #배경부분을 제외한 Cr의 히스토그램 평활화
                    equalizeCr = cv2.equalizeHist(Cr[Cr!=0])

                   
                    #히스토그램 = None 일 때
                    if equalizeY is None:  
                        continue

                    #Y, reshape    
                    Y[Y != 0] = equalizeY.reshape(equalizeY.shape[0])

                    #Cb, reshape
                    Cb[Cb!=0] = equalizeCb.reshape(equalizeCb.shape[0])

                    #Cr, reshape
                    Cr[Cr!=0] = equalizeCr.reshape(equalizeCr.shape[0])
                    
                    #이미지 합치기
                    img_ycbcr2 = cv2.merge([Y, Cb, Cr])

                    #YCrCb로 변경                     
                    HistImageResult = cv2.cvtColor(img_ycbcr2, cv2.COLOR_YCrCb2BGR)

                    #배경 흰색
                    HistImageResult[mask == 0] =255,255,255

                    #이미지 저장
                    cv2.imwrite(FileSavePathName, HistImageResult)
                    '''

                    #RGB -> YCrCb에 접근하기 위한 공식
                    '''
                    #이미지를 BGR로 읽는다
                    img_BGR = cv2.cvtColor(ImageResult,cv2.COLOR_RGB2BGR)

                    #B, G, R split
                    B, G, R = cv2.split(img_BGR)

                    #BGR -> YCrCb로 변경
                    Y = np.uint8((299*R + 587*G + 114*B)/1000)
                    Cb = np.uint8((0.5643*(B - Y) + 128))
                    Cr = np.uint8(0.7132*(R - Y) + 128)


                    #YCrCb 이미지 
                    img_ycbcr = cv2.merge([Y, Cb, Cr])

                    #Y2, Cb2, Cr2 split
                    Y2, Cb2, Cr2 = cv2.split(img_ycbcr)

                    #YCrCb -> BGR로 변경
                    R2 = 1000*Y2 + 1402*(Cr2-128)/1000
                    G2 = (1000*Y2 - 714*(Cr2-128) - 334*(Cb2-128))/1000
                    B2 = (1000*Y2 + 1772*(Cb2-128))/1000

                    #BGR 이미지
                    img_BGR2 = cv2.merge([B2, G2, R2])
                    '''                      


    #JSON으로 저장하기 
    def JsonSave(self, DictSaveName):

        #아이디, 이미지명
        for id, filename in self.jsonlist:

            #딕셔너리, id : 변환된 이미지명들
            self.key_dick[id].append(filename)

        #test1.json으로 저장 
        with open("{}.json".format(DictSaveName),"w") as test_json:
            js.dump(self.key_dick, test_json)
