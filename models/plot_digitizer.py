import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import sys

def main(image_path):
    try:
        img = mpimg.imread(image_path)
    except FileNotFoundError:
        print(f"❌ Không tìm thấy file ảnh: {image_path}")
        sys.exit(1)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.imshow(img)
    ax.axis('off')  # Ẩn trục tọa độ pixel của ảnh

    # ---------------------------------------------------------
    # 1. LẤY CÁC ĐIỂM CHUẨN (CALIBRATION)
    # ---------------------------------------------------------
    print("="*60)
    print("📍 BƯỚC 1: ĐỊNH CHUẨN TRỤC TỌA ĐỘ")
    print("Vui lòng click vào màn hình hình ảnh theo ĐÚNG thứ tự 4 điểm sau:")
    print("  1. Điểm bắt đầu của trục X (ví dụ: x = 0)")
    print("  2. Điểm kết thúc của trục X (ví dụ: x = 3)")
    print("  3. Điểm bắt đầu của trục Y (ví dụ: y = 0)")
    print("  4. Điểm cao nhất của trục Y (ví dụ: y = 1.4)")
    
    plt.title("Bước 1: Click 4 điểm chuẩn: X_min, X_max, Y_min, Y_max")
    
    # Lấy 4 điểm click từ người dùng
    refs = plt.ginput(4, timeout=-1, show_clicks=True)
    if len(refs) < 4:
        print("❌ Bạn chưa click đủ 4 điểm chuẩn. Đang thoát...")
        sys.exit(1)

    px_x_min, _ = refs[0]
    px_x_max, _ = refs[1]
    _, px_y_min = refs[2] # Lưu ý: Trong ảnh, Y pixel tăng từ trên xuống dưới
    _, px_y_max = refs[3]

    print("\nNhập giá trị thực tế của các điểm bạn vừa click:")
    val_x_min = float(input("  > Giá trị thực của X_min (vd 0): "))
    val_x_max = float(input("  > Giá trị thực của X_max (vd 3): "))
    val_y_min = float(input("  > Giá trị thực của Y_min (vd 0): "))
    val_y_max = float(input("  > Giá trị thực của Y_max (vd 1.4): "))

    # Hàm chuyển đổi pixel sang tọa độ thực
    def px_to_real(px, py):
        real_x = val_x_min + (px - px_x_min) * (val_x_max - val_x_min) / (px_x_max - px_x_min)
        real_y = val_y_min + (py - px_y_min) * (val_y_max - val_y_min) / (px_y_max - px_y_min)
        return real_x, real_y

    # ---------------------------------------------------------
    # 2. LẤY DỮ LIỆU THỰC NGHIỆM (DIGITIZING)
    # ---------------------------------------------------------
    print("\n" + "="*60)
    print("🎯 BƯỚC 2: TRÍCH XUẤT DỮ LIỆU")
    print("Click chuột TRÁI vào các chấm dữ liệu trên đồ thị.")
    print("Nhấn chuột GIỮA (nút cuộn) hoặc phím ENTER để kết thúc việc lấy điểm.")
    plt.title("Bước 2: Click vào các chấm dữ liệu. Nhấn Enter để hoàn tất.")
    
    data_points = plt.ginput(-1, timeout=-1, show_clicks=True, mouse_stop=2) # mouse_stop=2 là chuột giữa
    plt.close()

    if not data_points:
        print("❌ Không có điểm dữ liệu nào được chọn.")
        sys.exit(0)

    # Tính toán tọa độ thực
    rho_list = []
    v_list = []
    for px, py in data_points:
        real_x, real_y = px_to_real(px, py)
        rho_list.append(real_x)
        v_list.append(real_y)

    # ---------------------------------------------------------
    # 3. XUẤT KẾT QUẢ RA CODE PYTHON
    # ---------------------------------------------------------
    print("\n" + "="*60)
    print("✅ HOÀN TẤT! COPY ĐOẠN CODE DƯỚI ĐÂY VÀO FILE CỦA BẠN:")
    print("="*60)
    
    # Format string sao cho đẹp giống numpy array
    rho_str = ", ".join([f"{val:.2f}" for val in rho_list])
    v_str = ", ".join([f"{val:.2f}" for val in v_list])
    
    print("import numpy as np\n")
    print("EMPIRICAL_RHO = np.array([")
    print(f"    {rho_str}")
    print("])\n")
    print("EMPIRICAL_V = np.array([")
    print(f"    {v_str}")
    print("])\n")


if __name__ == "__main__":
    # Thay 'hinh_cua_ban.png' bằng đường dẫn tới ảnh bạn cắt từ bài báo
    image_file = "models/empirical_data.png"
    main(image_file)