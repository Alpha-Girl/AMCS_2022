from paddleocr import PaddleOCR, draw_ocr

# Paddleocr目前支持的多语言语种可以通过修改lang参数进行切换
# 例如`ch`, `en`, `fr`, `german`, `korean`, `japan`
ocr = PaddleOCR(use_angle_cls=True, lang="ch")  # need to run only once to download and load model into memory
img_path = "C:\\Users\\dell\\Desktop\\OIP.jpg"
result = ocr.ocr(img_path, cls=True)
# print("####")
# print(result)

for line in result:
    print(line)

# # 显示结果
# from PIL import Image

# image = Image.open(img_path).convert('RGB')
# boxes = [line[0] for line in result]
# print("####")
# print(boxes)

# txts = [line[1][0] for line in result]
# print("####")
# print(txts)
# print(type(txts))
# scores = [line[1][1] for line in result]
# print("####")
# print(scores)
# print(type(scores))
# im_show = draw_ocr(image, boxes, txts, scores, font_path='./fonts/simfang.ttf')
# im_show = Image.fromarray(im_show)
# im_show.save('result.jpg')

# draw result
from PIL import Image
image = Image.open(img_path).convert('RGB')
boxes = [detection[0] for line in result for detection in line] # Nested loop added
txts = [detection[1][0] for line in result for detection in line] # Nested loop added
scores = [detection[1][1] for line in result for detection in line] # Nested loop added
im_show = draw_ocr(image, boxes, txts, scores, font_path='./fonts/simfang.ttf')
im_show = Image.fromarray(im_show)
im_show.save('test.jpg')