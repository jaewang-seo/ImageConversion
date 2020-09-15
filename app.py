import process as pc

#명환 선배님 코드 
###########################################
#json 분할

# testJson = pc.ProcessJSON()

# testJson.jsonFileSplit('./Json','Cheonan.json') # 
# testJson.jsonFileSplitSave('./JsonSplit')
# jsonSplits = testJson.getJsonSplit()

# testJson.valueClear()

# testJson = pc.ProcessJSON()
# testJson.jsonFileMerge('./JsonSplit')
# testJson.jsonFileMergeSave('./JsonMerge', 'cocojson.json')  #위치,이름
# jsonMerges = testJson.getJsonMerge()


#클래스
koo = pc.IMAGECONVERSION()    

#process.py 사용
jsonfile = pc.ProcessJSON()

#json 불러오기 : jsonFileLoad(jsonFilePath, jsonFileName)
jsonfile.jsonFileLoad("C:/Users/Trip1/Desktop/Python/200707_emblem/.exports", "coco-1594624682.9582064.json")
json_data = jsonfile.json


#딕셔너리 세팅
koo.datasetting(json_data)

#이미지 변환하기 : ImageConvert(불러올 이미지 경로, 저장할 이미지 경로) 
koo.ImageConvert('C:/Users/Trip1/Desktop/Python/TEST/', 'C:/Users/Trip1/Desktop/python/SaveTEST2/')

#딕셔너리 만들기
koo.IdDict()

#이미지 보정
koo.histogram()

#JSON으로 저장하기 : JsonSave(저장할 Json명)
koo.JsonSave('test1')