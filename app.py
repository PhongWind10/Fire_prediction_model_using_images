import os
import gradio as gr
import numpy as np
import tensorflow as tf
from PIL import Image

# 1. ĐƯỜNG DẪN MÔ HÌNH
# Đảm bảo file mô hình của bạn nằm trong thư mục 'Model' cùng cấp với file app.py này
# Thay đổi 'model_cnn_fire.h5' thành tên file chính xác của bạn (ví dụ: .h5, .keras)
MODEL_PATH = os.path.join("Model", "model_cnn_fire.h5")

# Tải mô hình vào bộ nhớ
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("👉 Đã tải mô hình CNN thành công!")
except Exception as e:
    print(f"❌ Lỗi tải mô hình: {e}")
    print("Vui lòng kiểm tra lại tên file mô hình trong thư mục 'Model'.")


# 2. HÀM DỰ ĐOÁN (PREDICTION FUNCTION)
def predict_fire(input_image):
    if input_image is None:
        return "Vui lòng tải lên một hình ảnh."

    try:
        # Bước A: Tiền xử lý ảnh giống như khi huấn luyện (Train)
        # Thay đổi kích thước (target_size) theo cấu hình mô hình của bạn (thường là 150x150, 224x224 hoặc 256x256)
        target_size = (224, 224)
        img = input_image.resize(target_size)

        # Chuyển ảnh thành mảng Numpy và chuẩn hóa (Normalize) về đoạn [0, 1]
        img_array = np.array(img) / 255.0

        # Thêm chiều batch (Batch dimension) thành (1, 224, 224, 3)
        img_array = np.expand_dims(img_array, axis=0)

        # Bước B: Dự đoán từ mô hình
        prediction = model.predict(img_array)[0][0]

        # Bước C: Xử lý kết quả đầu ra dựa trên bài toán 3 biến / phân lớp của bạn
        # Giả sử mô hình dạng nhị phân (Sigmoid): gần 1 là Cháy, gần 0 là An toàn
        if prediction > 0.5:
            confidence = prediction * 100
            result = f"🚨 CẢNH BÁO: CÓ HỎA HOẠN DỰA TRÊN MÔ HÌNH CNN! (Độ chính xác: {confidence:.2f}%)"
        else:
            confidence = (1 - prediction) * 100
            result = f"✅ AN TOÀN: Không phát hiện đám cháy. (Độ chính xác: {confidence:.2f}%)"

        return result

    except Exception as e:
        return f"🚨 Đã xảy ra lỗi khi xử lý ảnh: {str(e)}"


# 3. XÂY DỰNG GIAO DIỆN GRADIO
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🧯 HỆ THỐNG NHẬN DIỆN ĐÁM CHÁY BẰNG MÔ HÌNH HỌC SÂU CNN
        ### Đồ án cơ sở - Hệ thống tự động phân tích và phát hiện hỏa hoạn qua hình ảnh.
        """
    )

    with gr.Row():
        with gr.Column():
            # Khung tải ảnh lên
            image_input = gr.Image(type="pil", label="Tải ảnh khu vực cần quét")
            submit_button = gr.Button("🔥 Phân Tích Hình Ảnh", variant="primary")

        with gr.Column():
            # Khung hiển thị kết quả
            text_output = gr.Textbox(
                label="Kết quả dự đoán từ AI", interactive=False, placeholder="Kết quả sẽ hiển thị tại đây..."
            )

    # Đặt sự kiện click nút hoặc submit ảnh
    submit_button.click(fn=predict_fire, inputs=image_input, outputs=text_output)

# 4. KHỞI CHẠY APP
if __name__ == "__main__":
    # Đặt share=False khi chạy trên Cloud (Hugging Face / Render)
    demo.launch()