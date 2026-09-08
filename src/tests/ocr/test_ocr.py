from paddleocr import PaddleOCR

ocr = PaddleOCR(
    lang="vi",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
)

image_path = "outputs/keyframes/K01_V001/K01_V001_s0001_k01.jpg"

result = ocr.predict(image_path)

print(result)