import os
import cv2


def detect_qr_code(image_path):
    """
    检测图片中是否包含二维码
    :param image_path: 图片路径
    :return: 布尔值（是否包含二维码）和解码结果列表
    """
    # 读取图片
    image = cv2.imread(image_path)
    if image is None:
        return False, []

    try:
        # 创建微信二维码检测器
        model_base_path = "models"  # 确保这个目录存在并包含所需模型文件
        detector = cv2.wechat_qrcode_WeChatQRCode(
            os.path.join(model_base_path, "detect.prototxt"),
            os.path.join(model_base_path, "detect.caffemodel"),
            os.path.join(model_base_path, "sr.prototxt"),
            os.path.join(model_base_path, "sr.caffemodel"),
        )

        # 图像预处理
        # 转换图像格式确保兼容性
        if len(image.shape) == 2:  # 如果是灰度图
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:  # 如果是RGBA
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

        # 使用微信二维码检测器进行检测
        decoded_text, points = detector.detectAndDecode(image)

        # 如果检测到二维码，在图片上标记
        if len(decoded_text) > 0:
            for i, text in enumerate(decoded_text):
                if points is not None and len(points) > i:
                    # 绘制二维码边界
                    pts = points[i].astype(int)
                    cv2.polylines(image, [pts], True, (0, 255, 0), 2)

                    # 在二维码上方显示解码文本
                    x = pts[0][0]
                    y = pts[0][1] - 10
                    cv2.putText(
                        image,
                        text,
                        (int(x), int(y)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                    )

            # 保存结果图片
            output_path = os.path.join("output", os.path.basename(image_path))
            try:
                success = cv2.imwrite(output_path, image)
                if success:
                    print(f"结果图片已保存到: {output_path}")
                else:
                    print(f"保存图片失败: {output_path}")
            except Exception as e:
                print(f"保存图片时发生错误: {str(e)}")

            # 创建与pyzbar兼容的返回结果
            class QRResult:
                def __init__(self, data, type="QRCODE"):
                    self.data = data.encode("utf-8")
                    self.type = type

            results = [QRResult(text) for text in decoded_text if text]
            return len(results) > 0, results

    except Exception as e:
        print(f"二维码检测过程中出错: {str(e)}")
        return False, []

    return False, []


# 使用示例
if __name__ == "__main__":
    image_dir = "input"  # 图片目录

    # 确保输出目录存在
    if not os.path.exists("output"):
        os.makedirs("output")

    # 获取目录下所有图片文件
    image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".gif")
    image_files = [
        f
        for f in os.listdir(image_dir)
        if os.path.isfile(os.path.join(image_dir, f))
        and f.lower().endswith(image_extensions)
    ]

    if not image_files:
        print("未找到任何图片文件。")
    else:
        print(f"找到 {len(image_files)} 个图片文件，开始处理...")

        for image_file in image_files:
            image_path = os.path.join(image_dir, image_file)
            print(f"\n处理图片: {image_file}")

            has_qr, results = detect_qr_code(image_path)

            if has_qr:
                print(f"在 {image_file} 中检测到二维码！")
                for qr in results:
                    print(f"二维码类型: {qr.type}")
                    print(f"二维码数据: {qr.data.decode('utf-8')}")
            else:
                print(f"在 {image_file} 中未检测到二维码。")

    model_files = [
        "detect.prototxt",
        "detect.caffemodel",
        "sr.prototxt",
        "sr.caffemodel",
    ]

    for file in model_files:
        path = os.path.join("models", file)
        if os.path.exists(path):
            print(f"✓ {file} 已找到")
        else:
            print(f"✗ {file} 未找到")
