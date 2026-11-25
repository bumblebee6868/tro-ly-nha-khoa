import os
import re
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta, timezone
from groq import Groq

# 1. CẤU HÌNH API KEY (Lấy từ Secrets của Streamlit)
try:
    API_KEY = st.secrets["GROQ_API_KEY"]
except Exception:
    API_KEY = os.environ.get("GROQ_API_KEY", "")

FILE_KHACH_HANG = "danh_sach_dat_lich.csv"

# 2. HÀM LƯU THÔNG TIN KHÁCH VÀO FILE CSV (CHUẨN GIỜ VIỆT NAM UTC+7)
def luu_lich_hen(ten_khach: str, so_dien_thoai: str, dich_vu: str, thoi_gian_hen: str):
    gio_vn = datetime.now(timezone(timedelta(hours=7))).strftime("%Y-%m-%d %H:%M")
    du_lieu_moi = {
        "Thời gian nhắn": gio_vn,
        "Tên khách": ten_khach.strip(),
        "Số điện thoại": so_dien_thoai.strip(),
        "Dịch vụ quan tâm": dich_vu.strip(),
        "Lịch hẹn mong muốn": thoi_gian_hen.strip()
    }
    df_moi = pd.DataFrame([du_lieu_moi])
    if os.path.exists(FILE_KHACH_HANG):
        df_moi.to_csv(FILE_KHACH_HANG, mode="a", header=False, index=False, encoding="utf-8-sig")
    else:
        df_moi.to_csv(FILE_KHACH_HANG, mode="w", header=True, index=False, encoding="utf-8-sig")

# 3. HÀM CHỌN MODEL THÔNG MINH NHẤT TRONG TÀI KHOẢN GROQ
def lay_danh_sach_model_thong_minh(client):
    tat_ca_model = [m.id for m in client.models.list().data]
    tu_khoa_loai = ["whisper", "guard", "tts", "orpheus", "allam", "distil"]
    model_hop_le = [
        m for m in tat_ca_model
        if not any(tk in m.lower() for tk in tu_khoa_loai)
    ]

    # Xếp hạng ưu tiên các model xử lý tiếng Việt & suy luận tốt nhất
    uu_tien = [
        "gpt-oss-120b",
        "llama-4-maverick",
        "llama-3.3-70b",
        "kimi-k2",
        "qwen3-32b",
        "llama-4-scout",
        "gpt-oss-20b",
        "llama-3.1-8b",
        "gemma2-9b",
    ]

    danh_sach_sap_xep = []
    for ten_uu_tien in uu_tien:
        for m in model_hop_le:
            if ten_uu_tien in m.lower() and m not in danh_sach_sap_xep:
                danh_sach_sap_xep.append(m)

    for m in model_hop_le:
        if m not in danh_sach_sap_xep:
            danh_sach_sap_xep.append(m)

    return danh_sach_sap_xep

# 4. ĐỌC DỮ LIỆU TIỆM & KỊCH BẢN TƯ VẤN TOÀN DIỆN
if os.path.exists("du_lieu_tiem.txt"):
    with open("du_lieu_tiem.txt", "r", encoding="utf-8") as f:
        noi_dung_tiem = f.read()
else:
    noi_dung_tiem = "Chưa có file du_lieu_tiem.txt."

SYSTEM_PROMPT = f"""
Bạn là chuyên viên tư vấn chăm sóc khách hàng chuyên nghiệp, tận tâm và thông minh của cơ sở dưới đây.

DỮ LIỆU CƠ SỞ & BẢNG GIÁ DỊCH VỤ:
{noi_dung_tiem}

NGUYÊN TẮC TƯ VẤN BẮT BUỘC:
1. NGÔN NGỮ & CHÍNH TẢ: Viết tiếng Việt chuẩn xác 100%, tự nhiên, lịch sự, mạch lạc, tuyệt đối không lặp từ hay dùng từ vô nghĩa. Xưng "em", gọi khách là "anh/chị" (hoặc gọi kèm tên nếu khách đã giới thiệu, ví dụ: "anh Huy").
2. TRUNG THỰC DỮ LIỆU: Chỉ tư vấn đúng các dịch vụ, mức giá, ưu đãi và giờ mở cửa có trong phần DỮ LIỆU CƠ SỞ ở trên. Tuyệt đối không tự bịa ra dịch vụ lạ hoặc giá sai lệch.
3. TRẢ LỜI TOÀN DIỆN MỌI TÌNH HUỐNG:
  - Khi khách hỏi "còn dịch vụ nào khác?": Liệt kê đầy đủ, rõ ràng toàn bộ các dịch vụ hiện có của cơ sở bằng gạch đầu dòng kèm mức giá để khách dễ chọn.
  - Khi khách phân vân hoặc chê giá cao: Tư vấn giải pháp phù hợp ngân sách, nhấn mạnh ưu đãi/bảo hành hiện có và mời khách qua thăm khám/soi da miễn phí trước.
  - Khi khách hỏi vấn đề chuyên môn sâu hoặc ngoài danh mục: Giải đáp ngắn gọn, dễ hiểu trong phạm vi kiến thức chung và khéo léo mời khách để lại SĐT để bác sĩ/chuyên viên gọi tư vấn trực tiếp.
4. QUY TRÌNH CHỐT LỊCH THÔNG MINH (RẤT QUAN TRỌNG):
  - Để chốt một lịch hẹn hoàn chỉnh cần 4 thông tin: (1) Tên khách, (2) Số điện thoại, (3) Dịch vụ quan tâm, (4) Thời gian muốn đến.
  - Nếu khách đã cho Tên và Giờ hẹn nhưng CHƯA có Số điện thoại: Hãy ghi nhớ Tên và Giờ đó, xác nhận lại với khách và hỏi xin thêm Số điện thoại (và dịch vụ nếu chưa rõ) để giữ lịch. KHÔNG được báo đã chốt xong khi chưa có Số điện thoại.
  - CHỈ KHI khách đã cung cấp Số điện thoại hợp lệ: Hãy xác nhận đầy đủ thông tin lịch hẹn, báo lễ tân sẽ liên hệ xác nhận trong 5 phút tới, và BẮT BUỘC thêm đúng 1 dòng thẻ kỹ thuật ở cuối cùng (không giải thích gì thêm về dòng này):
[CHOT_LICH: Tên khách | Số điện thoại | Dịch vụ | Thời gian hẹn]
Ví dụ: [CHOT_LICH: Anh Huy | 0987654321 | Tẩy trắng răng | 15h chiều mai]
"""

# 5. GIAO DIỆN WEB DEMO
st.set_page_config(page_title="Trợ lý AI Tư vấn & Đặt lịch", page_icon="💬", layout="wide")
st.title("💬 Trợ lý AI Tư vấn Khách hàng")
st.caption("Hỏi đáp tự nhiên mọi tình huống, nhớ ngữ cảnh hội thoại và tự động lưu lịch khi đủ SĐT.")

with st.sidebar:
    st.header("📋 Danh sách khách tự động chốt")
    if os.path.exists(FILE_KHACH_HANG):
        st.dataframe(pd.read_csv(FILE_KHACH_HANG, dtype=str), use_container_width=True)
    else:
        st.info("Chưa có khách đặt lịch. Hãy thử nhắn tên, dịch vụ, giờ hẹn và SĐT vào khung chat!")

    st.divider()
    if st.button("🔄 Làm mới cuộc trò chuyện", use_container_width=True):
        st.session_state.messages = [
            {"role": "assistant", "content": "Dạ em chào anh/chị ạ! Em có thể hỗ trợ tư vấn dịch vụ, báo giá hoặc đặt lịch ưu đãi nào cho mình hôm nay ạ?"}
        ]
        st.rerun()

    if "model_dang_chay" in st.session_state:
        st.caption(f"⚙️ AI đang dùng: `{st.session_state.model_dang_chay}`")

client = Groq(api_key=API_KEY)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Dạ em chào anh/chị ạ! Em có thể hỗ trợ tư vấn dịch vụ, báo giá hoặc đặt lịch ưu đãi nào cho mình hôm nay ạ?"}
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Nhập câu hỏi của khách hàng (VD: Bên em có những dịch vụ nào?)..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Đang tư vấn..."):
            try:
                danh_sach_model = lay_danh_sach_model_thong_minh(client)
                lich_su_gui = [{"role": "system", "content": SYSTEM_PROMPT}] + st.session_state.messages

                cau_tra_loi = ""
                loi_cuoi = None

                # Thử lần lượt các model thông minh nhất, tự động dự phòng nếu gặp giới hạn
                for model_id in danh_sach_model[:4]:
                    try:
                        response = client.chat.completions.create(
                            model=model_id,
                            messages=lich_su_gui,
                            temperature=0.3,
                            max_tokens=1024,
                            frequency_penalty=0.3,
                        )
                        noi_dung = response.choices[0].message.content or ""
                        noi_dung = re.sub(r"<think>.*?</think>", "", noi_dung, flags=re.DOTALL).strip()
                        if noi_dung:
                            cau_tra_loi = noi_dung
                            st.session_state.model_dang_chay = model_id
                            break
                    except Exception as err:
                        loi_cuoi = err
                        continue

                if not cau_tra_loi:
                    raise Exception(f"Không kết nối được mô hình AI. Chi tiết: {loi_cuoi}")

                # Tự động tách thẻ [CHOT_LICH: ...] và lưu vào file CSV
                match = re.search(r"\[CHOT_LICH:\s*(.*?)\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\]", cau_tra_loi)
                if match:
                    luu_lich_hen(match.group(1), match.group(2), match.group(3), match.group(4))
                    cau_tra_loi = re.sub(r"\[CHOT_LICH:.*?\]", "", cau_tra_loi).strip()

                st.markdown(cau_tra_loi)
                st.session_state.messages.append({"role": "assistant", "content": cau_tra_loi})
                st.rerun()
            except Exception as e:
                st.error(f"Lỗi kết nối API: {e}")
