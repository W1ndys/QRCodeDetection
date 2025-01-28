import os
import cv2
import numpy as np


def detect_qr_code(image_path):
    """
    检测图片中是否包含二维码
    :param image_path: 图片路径
    :return: 布尔值（是否包含二维码）和解码结果列表
    """
    # 读取图片
    try:
        image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"读取图片失败: {str(e)}")
        return False, []

    if image is None:
        print(f"无法读取图片: {image_path}")
        return False, []

    try:

        # 创建微信二维码检测器
        model_base_path = "models"  # 确保这个目录存在并包含所需模型文件
        detector = cv2.wechat_qrcode.WeChatQRCode(
            os.path.join(model_base_path, "detect.prototxt"),
            os.path.join(model_base_path, "detect.caffemodel"),
            os.path.join(model_base_path, "sr.prototxt"),
            os.path.join(model_base_path, "sr.caffemodel"),
        )

        # 图像预处理
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

        # 创建多个图像处理版本
        processed_images = []

        # 1. 原图（保持原图优先级最高）
        processed_images.append(image)

        # 2. 基础预处理（轻微增强）
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        denoised = cv2.GaussianBlur(gray, (3, 3), 0)
        _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary_rgb = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
        processed_images.append(binary_rgb)

        # 3. 自适应二值化（对复杂背景更好）
        adaptive_binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        adaptive_rgb = cv2.cvtColor(adaptive_binary, cv2.COLOR_GRAY2RGB)
        processed_images.append(adaptive_rgb)

        # 4. 轻微对比度增强
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))  # 降低clipLimit
        l = clahe.apply(l)
        enhanced_lab = cv2.merge([l, a, b])
        processed_images.append(cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB))

        # 5. 轻微锐化
        kernel = np.array(
            [[-0.5, -0.5, -0.5], [-0.5, 5, -0.5], [-0.5, -0.5, -0.5]]
        )  # 降低锐化强度
        sharpened = cv2.filter2D(image, -1, kernel)
        processed_images.append(sharpened)

        # 在所有处理后的图像上尝试检测
        decoded_text = []
        points = None
        for test_image in processed_images:
            current_decoded_text, current_points = detector.detectAndDecode(test_image)
            if len(current_decoded_text) > 0:
                # 如果检测到结果，保存结果并继续检测其他版本
                decoded_text.extend(current_decoded_text)
                if points is None or len(points) < len(current_points):
                    points = current_points
                    image = test_image

        # 去重结果
        decoded_text = list(set(decoded_text))

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


def check_image_quality(image):
    """检查图像质量"""
    # 检查亮度
    brightness = np.mean(image)
    if brightness < 30 or brightness > 225:
        return False

    # 检查对比度
    contrast = image.std()
    if contrast < 20:
        return False

    return True


# 使用示例
if __name__ == "__main__":
    # 定义颜色的ANSI转义码
    RED = "\033[91m"
    GREEN = "\033[92m"  # 添加绿色
    RESET = "\033[0m"

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
        print(f"{RED}未找到任何图片文件。{RESET}")
    else:
        print(f"{RED}找到 {len(image_files)} 个图片文件，开始处理...{RESET}")

        for image_file in image_files:
            image_path = os.path.join(image_dir, image_file)
            print(f"\n{RED}处理图片: {image_file}{RESET}")

            has_qr, results = detect_qr_code(image_path)

            if has_qr:
                print(f"{GREEN}在 {image_file} 中检测到二维码！{RESET}")
                for qr in results:
                    print(f"{GREEN}二维码类型: {qr.type}{RESET}")
                    print(f"{GREEN}二维码数据: {qr.data.decode('utf-8')}{RESET}")
            else:
                print(f"{RED}在 {image_file} 中未检测到二维码。{RESET}")
