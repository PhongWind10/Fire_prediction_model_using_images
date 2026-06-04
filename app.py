# app.py
import gradio as gr
import tensorflow as tf
import numpy as np
import cv2
import io
import datetime
import pandas as pd
import matplotlib.pyplot as plt

# Tải kiến trúc mô hình chạy thực nghiệm
model = globals().get('model', None)

if model is None:
    try:
        # LƯU Ý: Sửa lại đường dẫn này nếu không chạy trên Google Colab
        model = tf.keras.models.load_model('/content/drive/MyDrive/Do_an_co_so/Model/best_forest_fire_model.h5')
        print("[+] Đã kết nối tải tệp lưu vết mô hình dự phòng từ Drive.")
    except Exception as e:
        print(f"[!] Hệ thống không tìm thấy file cấu hình trọng số: {e}")
        model = None # Khai báo rõ là None để không bị NameError ở hàm bên dưới

detection_history_list = []

# Hàm sinh bản đồ nhiệt tối ưu hóa luồng tương thích bộ nhớ đệm
def make_gradcam_heatmap_gradio(img_array, model_instance, last_conv_layer_name, target_class_idx):
    grad_model_inputs = tf.keras.Input(shape=model_instance.input_shape[1:])
    x = grad_model_inputs
    last_conv_layer_output = None
    for layer in model_instance.layers:
        x = layer(x)
        if layer.name == last_conv_layer_name:
            last_conv_layer_output = x
    model_output_tensor = x
    grad_model = tf.keras.models.Model(inputs=grad_model_inputs, outputs=[last_conv_layer_output, model_output_tensor])
    with tf.GradientTape() as tape:
        last_conv_layer_output_val, preds = grad_model(img_array)
        class_channel = preds[:, target_class_idx]
    grads = tape.gradient(class_channel, last_conv_layer_output_val)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output_single_image = last_conv_layer_output_val[0]
    heatmap = tf.squeeze(last_conv_layer_output_single_image @ pooled_grads[..., tf.newaxis])
    heatmap = tf.maximum(heatmap, 0)
    return (heatmap / (tf.math.reduce_max(heatmap) + 1e-8)).numpy()

def overlay_heatmap_gradio(heatmap, original_image, alpha=0.5):
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    heatmap_colored = cv2.resize(heatmap_colored, (original_image.shape[1], original_image.shape[0]))
    return cv2.addWeighted(heatmap_colored, alpha, original_image, 1 - alpha, 0)

def get_last_conv_layer_name_gradio(model_instance):
    """
    Hàm tự động duyệt ngược các lớp của mô hình để tìm lớp tích chập cuối cùng
    """
    for layer in reversed(model_instance.layers):
        if 'conv' in layer.name:
            return layer.name
    return None

# Hàm sinh đồ thị cột độ tin cậy phân phối cho giao diện Gradio UI
def plot_confidence_bar_gradio(predictions):
    labels = ["CÓ CHÁY (FIRE)", "AN TOÀN (NON-FIRE)", "CÓ KHÓI (SMOKE)"]
    values = [float(predictions[0]), float(predictions[1]), float(predictions[2])]
    colors = [(1.0, 0.0, 0.0), (0.0, 0.5, 0.0), (0.5, 0.5, 0.5)]
    fig, ax = plt.subplots(figsize=(6, 3))
    bars = ax.barh(labels, values, color=colors, edgecolor='black')
    ax.set_xlim(0, 1)
    ax.set_xlabel("Mức xác suất")
    ax.invert_yaxis()
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.02, bar.get_y() + bar.get_height()/2, f'{width:.4f}', va='center', ha='left', color='black', fontweight='bold')
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    buf.seek(0)
    return (plt.imread(buf) * 255).astype(np.uint8)[:,:,:3]

def plot_raw_heatmaps_chart_gradio(hm_fire, hm_non_fire, hm_smoke):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(hm_fire, cmap='jet', vmin=0, vmax=1); axes[0].set_title("Heatmap 'Fire'"); axes[0].axis('off')
    axes[1].imshow(hm_non_fire, cmap='jet', vmin=0, vmax=1); axes[1].set_title("Heatmap 'Non-Fire'"); axes[1].axis('off')
    axes[2].imshow(hm_smoke, cmap='jet', vmin=0, vmax=1); axes[2].set_title("Heatmap 'Smoke'"); axes[2].axis('off')
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    buf.seek(0)
    return (plt.imread(buf) * 255).astype(np.uint8)[:,:,:3]

# Hàm tự động khoanh vùng khu vực nguy hiểm cục bộ (ROI - Bounding Boxes) dựa trên phân bổ năng lượng mạng tích chập
def draw_roi_on_image_gradio(original_image, heatmap_data, threshold=0.6, color=(255, 0, 0), label_text='DANGER'):
    heatmap_resized = cv2.resize(heatmap_data, (original_image.shape[1], original_image.shape[0]))
    binary_heatmap = (heatmap_resized > threshold).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary_heatmap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_with_roi = original_image.copy()
    for contour in contours:
        if cv2.contourArea(contour) > 80:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(img_with_roi, (x, y), (x + w, y + h), color, 3)
            cv2.putText(img_with_roi, label_text, (x, y - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    return img_with_roi

# Hàm dự toán logic trung tâm cho giao diện Gradio UI
def predict_fire_interface(image):
    if model is None or image is None:
        return {"Mô hình bị lỗi hệ thống": 0.0}, None, None, None, None, pd.DataFrame(detection_history_list)

    img_resized = cv2.resize(image, (128, 128))
    img_expanded = np.expand_dims(img_resized, axis=0)

    # Dự báo phân tích đa lớp kết quả
    predictions = model.predict(img_expanded)[0]
    class_ui_labels = ["CÓ CHÁY (FIRE)", "AN TOÀN (NON-FIRE)", "CÓ KHÓI (SMOKE)"]
    pred_idx = np.argmax(predictions)
    label_text_main = class_ui_labels[pred_idx]

    label_output = {class_ui_labels[i]: float(predictions[i]) for i in range(3)}
    confidence_plot_img = plot_confidence_bar_gradio(predictions)

    last_conv_layer = get_last_conv_layer_name_gradio(model)
    if last_conv_layer is None:
        return label_output, None, None, confidence_plot_img, image, pd.DataFrame(detection_history_list)

    try:
        hm_fire = make_gradcam_heatmap_gradio(img_expanded, model, last_conv_layer, target_class_idx=0)
        hm_non_fire = make_gradcam_heatmap_gradio(img_expanded, model, last_conv_layer, target_class_idx=1)
        hm_smoke = make_gradcam_heatmap_gradio(img_expanded, model, last_conv_layer, target_class_idx=2)

        overlay_fire = overlay_heatmap_gradio(hm_fire, image)
        overlay_non_fire = overlay_heatmap_gradio(hm_non_fire, image)
        overlay_smoke = overlay_heatmap_gradio(hm_smoke, image)

        combined_heatmap_plot = plot_raw_heatmaps_chart_gradio(hm_fire, hm_non_fire, hm_smoke)
        combined_raw_image = np.hstack((overlay_fire, overlay_non_fire, overlay_smoke))

        roi_image = image.copy()
        if pred_idx == 0:
            roi_image = draw_roi_on_image_gradio(roi_image, hm_fire, threshold=0.6, color=(255, 0, 0), label_text='FIRE AREA')
        elif pred_idx == 2:
            roi_image = draw_roi_on_image_gradio(roi_image, hm_smoke, threshold=0.6, color=(128, 128, 128), label_text='SMOKE AREA')
    except Exception as e:
        print(f"[!] Lỗi biên dịch xử lý luồng Heatmap UI: {e}")
        combined_heatmap_plot, combined_raw_image, roi_image = None, None, image

    # Thiết lập nhật ký thời gian thực thực thi hệ thống kiểm soát thông minh
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = {
        "Thời gian kích hoạt": timestamp,
        "Chẩn đoán quyết định": label_text_main,
        "Xác suất Lửa (Fire)": f"{predictions[0]:.4f}",
        "Xác suất Khói (Smoke)": f"{predictions[2]:.4f}",
        "Cảnh báo Nguy hiểm?": "BÁO ĐỘNG ĐỎ" if (pred_idx == 0 or pred_idx == 2) else "AN TOÀN"
    }
    detection_history_list.insert(0, log_entry)
    if len(detection_history_list) > 10:
        detection_history_list.pop()

    return label_output, combined_heatmap_plot, combined_raw_image, confidence_plot_img, roi_image, pd.DataFrame(detection_history_list)

# Tạo lập khung giao diện UI Đồ án Khoa học dữ liệu cao cấp bằng Gradio Blocks
interface = gr.Interface(
    fn=predict_fire_interface,
    inputs=gr.Image(label="Tải tài nguyên hình ảnh hiện trường đám cháy lên hệ thống"),
    outputs=[
        gr.Label(num_top_classes=3, label="Phân tích xác suất nhận diện (Mô hình toán học Đa lớp)"),
        gr.Image(label="Hệ thống ma trận Bản đồ nhiệt phân lớp đối sánh (Raw Heatmaps)", type="numpy"),
        gr.Image(label="Ảnh tích hợp chồng lớp định vị không gian (Grad-CAM Overlays Side-by-Side)", type="numpy"),
        gr.Image(label="Trực quan biểu đồ cột mức độ tin cậy quyết định dự phòng", type="numpy"),
        gr.Image(label="Tự động phát hiện khoanh vùng đối tượng đám cháy nguy hiểm (ROI Bounding Boxes)", type="numpy"),
        gr.Dataframe(label="Hệ thống cơ sở dữ liệu Nhật ký lịch sử phát hiện đám cháy thời gian thực")
    ],
    title="🔥 HỆ THỐNG PHÁT HIỆN ĐÁM CHÁY & KHÓI RỪNG BẰNG MÔ HÌNH HỌC SÂU CNN CÓ GIẢI THÍCH (HEATMAP XAI DEMO)",
    description="Hệ thống tích hợp thuật toán CNN trích xuất đặc trưng hình học kết hợp phương pháp giải thích Grad-CAM nhằm khoanh vùng định vị tọa độ phân bổ năng lượng đám cháy, hỗ trợ tối đa công tác cứu nạn.",
    theme="default"
)

if __name__ == "__main__":
    # Khởi chạy hệ thống liên kết máy chủ ảo phát hành đường link chia sẻ trực tuyến (Public Share Link)
    interface.launch(share=True, debug=True)