import streamlit as st
import webbrowser

# Set judul aplikasi
st.title("Alat Penyiraman dan Pemupukan Otomatis Tanaman Cabai")

# Membuat form untuk input alamat IP
ip_address = st.text_input("IP Server", "http://")


if st.button("Masuk"):
    if ip_address:
        if not ip_address.startswith("http://") and not ip_address.startswith("https://"):
            ip_address = "http://" + ip_address

        webbrowser.open(ip_address)
        st.success(f"Redirecting to: {ip_address}")
    else:
        st.error("Masukkan alamat IP server yang valid")
