# invoice_generator_20laundry.py
import streamlit as st
import pandas as pd
from datetime import datetime
import io
import matplotlib.pyplot as plt

# ========== Helpers & Backend ==========

def _parse_tanggal(s):
    """Parse 'Tanggal' like '30/09/2025 22:04:37' -> datetime."""
    for fmt in ("%d/%m/%Y %H:%M:%S", "%d/%m/%Y"):
        try:
            return datetime.strptime(str(s), fmt)
        except Exception:
            pass
    return pd.NaT

def _clean_order_no(s):
    """Clean values like =\"20250930-12302\" -> 20250930-12302."""
    s = str(s)
    return s[2:-1] if s.startswith('="') and s.endswith('"') else s

def format_rp(value: float, decimals: int = 2) -> str:
    """Indonesian Rupiah string with chosen decimals."""
    v = 0.0 if value is None or value != value else float(value)  # handle NaN
    s = f"Rp{v:,.{decimals}f}"            # 1,234,567.89
    return s.replace(",", "X").replace(".", ",").replace("X", ".")  # Rp1.234.567,89

def load_rebill_df(file, skiprows=5) -> pd.DataFrame:
    """Load CSV, normalize columns, parse date & totals."""
    df = pd.read_csv(file, skiprows=skiprows, engine="python")
    df.columns = [str(c).strip() for c in df.columns]

    # Order number cleanup (first matching column)
    for c in df.columns:
        if c.lower() in ("no", "no nota", "nota", "no.", "id"):
            df[c] = df[c].map(_clean_order_no)
            break

    if "Tanggal" in df.columns:
        df["Tanggal_dt"] = df["Tanggal"].map(_parse_tanggal)
    if "Total" in df.columns:
        df["Total_num"] = pd.to_numeric(df["Total"], errors="coerce")
    return df

def normalize_key_lookup(mapping: dict, key: str) -> str:
    """Return exact mapping key that matches case-insensitively, else original key."""
    key_lower = key.strip().lower()
    for k in mapping.keys():
        if k.strip().lower() == key_lower:
            return k
    return key

def build_invoice_dict(df: pd.DataFrame, agents_dict: dict, agent_name: str,
                       month_year: str, discount_rate: float = 0.2) -> dict:
    """Build invoice dict {rows, totals} for given agent and month."""
    no_col = next((c for c in df.columns if c.lower() in ("no", "no nota", "nota", "no.", "id")), None)
    if not no_col:
        raise ValueError("Could not find a 'No/No Nota' column.")
    if "Nama Pelanggan" not in df.columns:
        raise ValueError("Column 'Nama Pelanggan' not found.")
    if "Tanggal_dt" not in df.columns:
        raise ValueError("Tanggal must be parsed (use load_rebill_df first).")

    total_col = "Total_num" if "Total_num" in df.columns else "Total"

    # Filter month
    year_i, month_i = map(int, month_year.split("-"))
    msk_month = (df["Tanggal_dt"].dt.year == year_i) & (df["Tanggal_dt"].dt.month == month_i)

    # Case-insensitive agent & customers
    agent_key = normalize_key_lookup(agents_dict, agent_name)
    customers = set(n.strip().lower() for n in agents_dict.get(agent_key, []))

    # Case-insensitive filter by customer
    df_sel = df[
        msk_month &
        df["Nama Pelanggan"].astype(str).str.strip().str.lower().isin(customers)
    ].copy()

    # Compute values (numeric)
    df_sel["Price"] = pd.to_numeric(df_sel[total_col], errors="coerce").fillna(0.0)
    df_sel["Discount"] = df_sel["Price"] * float(discount_rate)
    df_sel["Amount"] = df_sel["Price"] - df_sel["Discount"]

    # Rows (keep numeric; format later)
    rows = []
    for _, r in df_sel.iterrows():
        rows.append({
            "Tanggal": r["Tanggal_dt"].strftime("%d/%m/%Y"),
            "No Nota": r[no_col],
            "Nama Pelanggan": r["Nama Pelanggan"],
            "Keterangan": agent_key.upper().replace(" ", ""),
            "Price": float(r["Price"]),
            "Discount Agen (Rp)": float(r["Discount"]),
            "Amount (Rp)": float(r["Amount"]),
            "Status": r.get("Status", "")
        })

    totals = {
        "Price": float(sum(x["Price"] for x in rows)),
        "Discount Agen (Rp)": float(sum(x["Discount Agen (Rp)"] for x in rows)),
        "Amount (Rp)": float(sum(x["Amount (Rp)"] for x in rows))
    }
    return {"rows": rows, "totals": totals}

def df_to_image(df: pd.DataFrame, title="Invoice Preview") -> io.BytesIO:
    """Render a DataFrame to PNG; bold header + last row."""
    fig_height = max(3.5, 0.6 + len(df) * 0.32)
    fig, ax = plt.subplots(figsize=(11, fig_height))
    ax.axis("off")

    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc="center",
        loc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7)

    # Bold header (row 0)
    for c in range(len(df.columns)):
        table[(0, c)].set_text_props(fontweight="bold")

    # Bold TOTAL row (last row in table coords = len(df))
    last_row_idx = len(df)  # header is 0; data start at 1
    for c in range(len(df.columns)):
        cell = table[(last_row_idx, c)]
        cell.set_text_props(fontweight="bold")
        cell.set_linewidth(1.6)
        cell.set_edgecolor("black")
        cell.set_facecolor("#f0f0f0")

    # Thicker borders for header & total
    for (r, c), cell in table.get_celld().items():
        if r in (0, last_row_idx):
            cell.set_linewidth(1.6)
            cell.set_edgecolor("black")

    plt.title(title, fontsize=20, fontweight="bold", pad=20)

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=220)
    plt.close(fig)
    buf.seek(0)
    return buf

def ensure_session_agents(default_agents: dict):
    """Initialize agents dict in session_state (persist while app runs)."""
    if "agents" not in st.session_state:
        st.session_state["agents"] = default_agents.copy()

def _split_names(text: str):
    """Split names from textarea: supports lines and commas, trims blanks."""
    parts = []
    for line in text.splitlines():
        parts += [p.strip() for p in line.split(",")]
    return [p for p in parts if p]

# ========== Streamlit UI ==========

st.set_page_config(page_title="20 Laundry Invoice Generator", layout="wide")
st.title("🧾 20 LAUNDRY - Invoice Generator")

uploaded_file = st.file_uploader("📂 Upload data ReBill CSV disini", type=["csv"])

if uploaded_file:
    df = load_rebill_df(uploaded_file, skiprows=5)
    st.success("✅ CSV berhasil terupload.")

    # Build customer choices from CSV
    all_customers = sorted(df["Nama Pelanggan"].astype(str).str.strip().dropna().unique())

    # Default agents; copied to session_state once
    default_agents = {
        "Pak Uus": [
            "Pak Roy RDP", "Bu Heni PU", "Daliyo RDP", "Bu Rini RDP", "Awan Pak Uus",
            "Pa Paryono RDP", "Novi RDP", "Pak Filbert RDP", "Pak Febert RDP", "Ibu Deki RDP",
            "Pak Waliyudinden", "Pa Joko RDP", "Bu Keke RDP", "Ipang RDP",
            "Pak Uus/ teh ita", "Foresh Hils B2", "Ibu Warti RDP", "Villa Bata Merah"
        ],
        "Harmoni (Mama Ola)": [
            "Umi Hani", "Ayman", "Bu Yuyu Rahayu", "Fahima", "Ibu Kartika", "Tante Kartika"
        ],
        "Aep Ciburial": [
            "Villa philanto RDP", "Villa Aito RDP", "Pa Aep", "Villa Forest Hill"
        ],
        "Bu Emi (UNISBA)": ["Bu Rini", "Bu Emi"],
        "SUKAVILLA": ["SUKAVILLA"]
    }

    ensure_session_agents(default_agents)

    st.subheader("🧍 Pengaturan Agent")

    colA, _ = st.columns(2)
    with colA:
        mode = st.radio("Tambahkan atau Pilih Agent", ["Pilih Agent tersedia", "Tambahkan Agent Baru"], horizontal=True)

    # ---------- ADD NEW (centered layout) ----------
    if mode == "Tambahkan Agent Baru":
        _left, _center, _right = st.columns([1, 2, 1])

        with _center:
            new_agent = st.text_input(
                "Masukan Nama Agent:",
                key="new_agent",
            ).strip()

            picked_customers = st.multiselect(
                "Pilih Nama Customer (bisa lebih dari 1)",
                options=all_customers,
                default=[],
                key="picked_from_csv",
            )

            extra_customers_text = st.text_area(
                "Lainnya",
                value="",
                placeholder="e.g.\nSUKAVILLA\nPak Roy RDP\nBu Rini",
                key="extra_customers_text",
                height=120,
            )

            extra_customers = [
                p.strip()
                for line in extra_customers_text.splitlines()
                for p in line.split(",")
                if p.strip()
            ]

            if st.button("➕ Tambahkan Agent", use_container_width=True):
                if not new_agent:
                    st.warning("Masukan Nama Agent.")
                else:
                    existing_key = normalize_key_lookup(st.session_state.agents, new_agent)
                    final_key = existing_key if existing_key in st.session_state.agents else new_agent
                    base = set(n.strip() for n in st.session_state.agents.get(final_key, []))
                    base.update(picked_customers)
                    base.update(extra_customers)
                    st.session_state.agents[final_key] = sorted(base, key=str.lower)
                    st.success(
                        f"Agent '{final_key}' sekarang memiliki "
                        f"{len(st.session_state.agents[final_key])} customers."
                    )

        agent_name = new_agent

    # ---------- SELECT EXISTING (show & edit customers) ----------
    else:
        agent_name = st.selectbox("👤 Pilih Agent", list(st.session_state.agents.keys()))
        current_key = normalize_key_lookup(st.session_state.agents, agent_name)
        current_customers = [
            c.strip()
            for c in st.session_state.agents.get(current_key, [])
            if str(c).strip()
        ]

        options_customers = sorted(
            set(all_customers) | set(current_customers),
            key=str.lower
        )

        st.markdown("**Customers untuk Agent ini:**")
        edited_customers = st.multiselect(
            "Ketik Nama Customer",
            options=options_customers,
            default=sorted(current_customers, key=str.lower),
            key="cust_multiselect"
        )

        typed_new_text = st.text_area(
            "Customer Lainnya",
            key="cust_text"
        )
        typed_new = _split_names(typed_new_text)

        if st.button("💾 Simpan Customer untuk Agent ini"):
            base = set(n.strip() for n in edited_customers)
            base.update(typed_new)
            base.update([c for c in current_customers if c not in all_customers])
            st.session_state.agents[current_key] = sorted(base, key=str.lower)
            st.success(
                f"Menyimpan {len(st.session_state.agents[current_key])} "
                f"customers untuk '{current_key}'."
            )

    month_year = st.text_input("📅 Bulan (YYYY-MM)", value=datetime.now().strftime("%Y-%m"))
    discount = st.slider("💸 Diskon", 0.0, 0.5, 0.20, 0.05)

    # Invoice date input
    invoice_date = st.date_input(
        "🗓️ Tanggal Invoice",
        value=datetime.now().date()
    )
    invoice_date_str = invoice_date.strftime("%d/%m/%Y")

    if st.button("Buat Invoice"):
        if not agent_name:
            st.warning("⚠️ Silahkan Pilih atau Tambah Agent terlebih dahulu.")
        else:
            invoice = build_invoice_dict(df, st.session_state.agents, agent_name, month_year, discount)

            if not invoice["rows"]:
                st.warning("⚠️ Nama Agent dan Customers tidak ditemukan.")
            else:
                inv_df = pd.DataFrame(invoice["rows"])
                inv_df["Price"] = inv_df["Price"].map(lambda x: format_rp(x, 0))
                inv_df["Discount Agen (Rp)"] = inv_df["Discount Agen (Rp)"].map(lambda x: format_rp(x, 0))
                inv_df["Amount (Rp)"] = inv_df["Amount (Rp)"].map(lambda x: format_rp(x, 0))

                total_row = {
                    "Tanggal": "",
                    "No Nota": "",
                    "Nama Pelanggan": "",
                    "Keterangan": "TOTAL",
                    "Price": format_rp(invoice["totals"]["Price"], 0),
                    "Discount Agen (Rp)": format_rp(invoice["totals"]["Discount Agen (Rp)"], 0),
                    "Amount (Rp)": format_rp(invoice["totals"]["Amount (Rp)"], 0),
                    "Status": ""
                }
                inv_df = pd.concat([inv_df, pd.DataFrame([total_row])], ignore_index=True)

                agent_key_display = normalize_key_lookup(st.session_state.agents, agent_name)

                st.subheader(f"📊 Invoice for {agent_key_display} — {month_year}")
                st.markdown(f"**Tanggal Invoice:** {invoice_date_str}")
                st.dataframe(inv_df, use_container_width=True)

                # -------- PNG download --------
                img_title = (
                    f"Invoice - {agent_key_display} ({month_year})\n"
                    f"Tanggal Invoice: {invoice_date_str}"
                )
                img_buf = df_to_image(inv_df, title=img_title)
                st.image(img_buf, caption="Invoice Preview", use_container_width=True)
                st.download_button(
                    label="⬇️ Download Invoice Image (PNG)",
                    data=img_buf,
                    file_name=f"Invoice_{agent_key_display}_{month_year}.png",
                    mime="image/png"
                )

                # -------- Excel download --------
                excel_buf = io.BytesIO()
                with pd.ExcelWriter(excel_buf, engine="xlsxwriter") as writer:
                    inv_df.to_excel(writer, index=False, sheet_name="Invoice", startrow=3)
                    wb = writer.book
                    ws = writer.sheets["Invoice"]

                    ws.write(0, 0, f"Invoice - {agent_key_display}")
                    ws.write(1, 0, f"Bulan: {month_year}")
                    ws.write(2, 0, f"Tanggal Invoice: {invoice_date_str}")

                    ws.set_column("A:A", 12)
                    ws.set_column("B:B", 18)
                    ws.set_column("C:C", 22)
                    ws.set_column("D:D", 14)
                    ws.set_column("E:G", 18)
                    ws.set_column("H:H", 10)

                excel_buf.seek(0)
                st.download_button(
                    label="⬇️ Download Invoice (Excel)",
                    data=excel_buf,
                    file_name=f"Invoice_{agent_key_display}_{month_year}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
