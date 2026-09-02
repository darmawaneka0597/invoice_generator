"""
20 LAUNDRY - Invoice Generator
Streamlit app for generating agent invoices from ReBill CSV exports.
Supports both legacy (Tanggal / No Nota) and current (Date / Bill ID) ReBill formats.
"""
import io
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
# Column name variants recognised from ReBill exports
_ID_COLS    = {"no", "no nota", "nota", "no.", "id", "bill id"}
_DATE_COLS  = {"tanggal", "date"}
_STATUS_DELETED_PREFIX = "deleted"
# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def _parse_date(value) -> "pd.Timestamp | pd.NaT":
    """Parse date strings like '30/06/2026 21:25:44' or '30/06/2026'."""
    for fmt in ("%d/%m/%Y %H:%M:%S", "%d/%m/%Y"):
        try:
            return datetime.strptime(str(value).strip(), fmt)
        except ValueError:
            pass
    return pd.NaT
def _clean_order_no(value: str) -> str:
    """Strip Excel formula wrappers: =\"20250930-12302\" -> 20250930-12302."""
    s = str(value)
    return s[2:-1] if s.startswith('="') and s.endswith('"') else s
def format_rp(value: float, decimals: int = 0) -> str:
    """Format a number as Indonesian Rupiah: Rp1.234.567"""
    v = 0.0 if (value is None or value != value) else float(value)
    raw = f"Rp{v:,.{decimals}f}"
    return raw.replace(",", "X").replace(".", ",").replace("X", ".")
def normalize_key(mapping: dict, key: str) -> str:
    """Return the dict key that case-insensitively matches *key*, or *key* itself."""
    key_lower = key.strip().lower()
    for k in mapping:
        if k.strip().lower() == key_lower:
            return k
    return key
def split_names(text: str) -> list[str]:
    """Split a textarea value on newlines and commas; strip blanks."""
    parts = []
    for line in text.splitlines():
        parts += [p.strip() for p in line.split(",")]
    return [p for p in parts if p]
# ─────────────────────────────────────────────
# DATA LAYER
# ─────────────────────────────────────────────
def load_rebill_df(file, skiprows: int = 5) -> pd.DataFrame:
    """
    Load a ReBill CSV export into a normalised DataFrame.
    - Supports both 'Tanggal'/'Date' date columns.
    - Supports both 'No Nota'/'Bill ID' order-number columns.
    - Drops rows with Status starting with 'Deleted'.
    - Parses date into a unified 'Tanggal_dt' column.
    - Parses Total into a numeric 'Total_num' column.
    """
    df = pd.read_csv(file, skiprows=skiprows, engine="python")
    df.columns = [str(c).strip() for c in df.columns]
    # Clean order-number column (handles Excel formula wrappers)
    id_col = next((c for c in df.columns if c.lower() in _ID_COLS), None)
    if id_col:
        df[id_col] = df[id_col].map(_clean_order_no)
    # Parse date -> unified Tanggal_dt
    date_col = next((c for c in df.columns if c.lower() in _DATE_COLS), None)
    if date_col:
        df["Tanggal_dt"] = df[date_col].map(_parse_date)
    # Parse total -> numeric
    if "Total" in df.columns:
        df["Total_num"] = pd.to_numeric(df["Total"], errors="coerce")
    # Drop deleted rows
    if "Status" in df.columns:
        mask_deleted = (
            df["Status"]
            .astype(str)
            .str.strip()
            .str.lower()
            .str.startswith(_STATUS_DELETED_PREFIX)
        )
        df = df[~mask_deleted].copy()
    return df
def default_month_from_data(df: pd.DataFrame) -> str:
    """
    Pick a sensible default 'YYYY-MM' for the period field based on the
    uploaded data itself, instead of today's date.

    Using datetime.now() as the default silently produces an empty invoice
    whenever the CSV covers a different month than "now" (e.g. exporting
    August data on September 1st) - the UI then reports "no transactions"
    for every agent, which looks like a customer-matching bug but isn't.

    We default to the most COMMON year-month in the data (mode), which is
    more robust than the max date in case the export includes a few
    stray rows from an adjacent month.
    """
    if "Tanggal_dt" in df.columns and df["Tanggal_dt"].notna().any():
        periods = df["Tanggal_dt"].dropna().dt.strftime("%Y-%m")
        if not periods.empty:
            return periods.mode().iloc[0]
    return datetime.now().strftime("%Y-%m")
def build_invoice(
    df: pd.DataFrame,
    agents: dict,
    agent_name: str,
    month_year: str,
    discount_rate: float = 0.20,
) -> dict:
    """
    Build invoice data for *agent_name* in *month_year* (format: 'YYYY-MM').
    Returns:
        {
            "rows": list[dict],          # one dict per transaction (numeric values)
            "totals": dict               # Price, Discount, Amount totals
        }
    Raises ValueError if required columns are missing.
    """
    # Resolve columns
    id_col = next((c for c in df.columns if c.lower() in _ID_COLS), None)
    if not id_col:
        raise ValueError(
            "Bill ID column not found. "
            f"Expected one of: {sorted(_ID_COLS)}. "
            f"Found: {list(df.columns)}"
        )
    if "Customer Name" not in df.columns:
        raise ValueError("Column 'Customer Name' not found in CSV.")
    if "Tanggal_dt" not in df.columns:
        raise ValueError(
            "Date column not found. "
            f"Expected one of: {sorted(_DATE_COLS)}."
        )
    total_col = "Total_num" if "Total_num" in df.columns else "Total"
    # Month filter
    year_i, month_i = map(int, month_year.split("-"))
    mask_month = (
        (df["Tanggal_dt"].dt.year == year_i) &
        (df["Tanggal_dt"].dt.month == month_i)
    )
    # Agent / customer filter (case-insensitive)
    agent_key = normalize_key(agents, agent_name)
    customers = {n.strip().lower() for n in agents.get(agent_key, [])}
    df_sel = df[
        mask_month &
        df["Customer Name"].astype(str).str.strip().str.lower().isin(customers)
    ].copy()
    # Compute columns
    df_sel["_price"]    = pd.to_numeric(df_sel[total_col], errors="coerce").fillna(0.0)
    df_sel["_discount"] = df_sel["_price"] * float(discount_rate)
    df_sel["_amount"]   = df_sel["_price"] - df_sel["_discount"]
    rows = []
    for _, r in df_sel.iterrows():
        rows.append({
            "Tanggal":            r["Tanggal_dt"].strftime("%d/%m/%Y"),
            "No Nota":            r[id_col],
            "Customer Name":      str(r["Customer Name"]).strip(),
            "Keterangan":         agent_key.upper().replace(" ", ""),
            "Price":              float(r["_price"]),
            "Discount Agen (Rp)": float(r["_discount"]),
            "Amount (Rp)":        float(r["_amount"]),
            "Status":             str(r.get("Status", "")),
        })
    totals = {
        "Price":              sum(x["Price"]              for x in rows),
        "Discount Agen (Rp)": sum(x["Discount Agen (Rp)"] for x in rows),
        "Amount (Rp)":        sum(x["Amount (Rp)"]        for x in rows),
    }
    return {"rows": rows, "totals": totals, "agent_key": agent_key}
# ─────────────────────────────────────────────
# EXPORT HELPERS
# ─────────────────────────────────────────────
def invoice_to_display_df(invoice: dict) -> pd.DataFrame:
    """Convert invoice dict to a formatted DataFrame (with TOTAL row)."""
    rows = [r.copy() for r in invoice["rows"]]
    for r in rows:
        r["Price"]              = format_rp(r["Price"])
        r["Discount Agen (Rp)"] = format_rp(r["Discount Agen (Rp)"])
        r["Amount (Rp)"]        = format_rp(r["Amount (Rp)"])
    total_row = {
        "Tanggal":            "",
        "No Nota":            "",
        "Customer Name":      "",
        "Keterangan":         "TOTAL",
        "Price":              format_rp(invoice["totals"]["Price"]),
        "Discount Agen (Rp)": format_rp(invoice["totals"]["Discount Agen (Rp)"]),
        "Amount (Rp)":        format_rp(invoice["totals"]["Amount (Rp)"]),
        "Status":             "",
    }
    rows.append(total_row)
    return pd.DataFrame(rows)
def render_invoice_png(display_df: pd.DataFrame, title: str) -> io.BytesIO:
    """Render a formatted invoice DataFrame to a PNG image."""
    fig_h = max(3.5, 0.55 + len(display_df) * 0.30)
    fig, ax = plt.subplots(figsize=(12, fig_h))
    ax.axis("off")
    table = ax.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    n_rows = len(display_df)
    for col in range(len(display_df.columns)):
        # Header row (index 0)
        table[(0, col)].set_text_props(fontweight="bold")
        table[(0, col)].set_facecolor("#1e3a5f")
        table[(0, col)].set_text_props(color="white", fontweight="bold")
        # Total row (last data row)
        cell = table[(n_rows, col)]
        cell.set_text_props(fontweight="bold")
        cell.set_facecolor("#e8f0fe")
        cell.set_linewidth(1.6)
        cell.set_edgecolor("#1e3a5f")
    # Alternate row shading
    for row in range(1, n_rows):
        bg = "#f7f9fc" if row % 2 == 0 else "white"
        for col in range(len(display_df.columns)):
            table[(row, col)].set_facecolor(bg)
    plt.title(title, fontsize=14, fontweight="bold", pad=16, color="#1e3a5f")
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=220)
    plt.close(fig)
    buf.seek(0)
    return buf
def render_invoice_excel(display_df: pd.DataFrame, agent_key: str, month_year: str, invoice_date_str: str) -> io.BytesIO:
    """Write the invoice to an Excel workbook and return a BytesIO buffer."""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        display_df.to_excel(writer, index=False, sheet_name="Invoice", startrow=4)
        wb = writer.book
        ws = writer.sheets["Invoice"]
        # Formats
        title_fmt  = wb.add_format({"bold": True, "font_size": 14, "font_color": "#1e3a5f"})
        meta_fmt   = wb.add_format({"font_size": 10, "font_color": "#555555"})
        header_fmt = wb.add_format({
            "bold": True, "bg_color": "#1e3a5f", "font_color": "white",
            "border": 1, "align": "center",
        })
        total_fmt  = wb.add_format({
            "bold": True, "bg_color": "#e8f0fe", "border": 1, "align": "center",
        })
        even_fmt   = wb.add_format({"bg_color": "#f7f9fc", "border": 1})
        odd_fmt    = wb.add_format({"border": 1})
        # Header rows
        ws.write(0, 0, f"20 LAUNDRY - Invoice Agent", title_fmt)
        ws.write(1, 0, f"Agent    : {agent_key}", meta_fmt)
        ws.write(2, 0, f"Periode  : {month_year}", meta_fmt)
        ws.write(3, 0, f"Tgl Invoice: {invoice_date_str}", meta_fmt)
        # Column headers at row 4
        for col_i, col_name in enumerate(display_df.columns):
            ws.write(4, col_i, col_name, header_fmt)
        # Data rows
        for row_i, row_data in enumerate(display_df.values):
            fmt = total_fmt if row_i == len(display_df) - 1 else (even_fmt if row_i % 2 == 0 else odd_fmt)
            for col_i, val in enumerate(row_data):
                ws.write(5 + row_i, col_i, val, fmt)
        # Column widths
        ws.set_column("A:A", 12)   # Tanggal
        ws.set_column("B:B", 20)   # No Nota
        ws.set_column("C:C", 24)   # Customer Name
        ws.set_column("D:D", 16)   # Keterangan
        ws.set_column("E:G", 20)   # Price / Discount / Amount
        ws.set_column("H:H", 30)   # Status
    buf.seek(0)
    return buf
# ─────────────────────────────────────────────
# DEFAULT AGENTS
# ─────────────────────────────────────────────
DEFAULT_AGENTS: dict[str, list[str]] = {
    "Pak Uus": [
        "Pak Roy RDP", "Bu Heni PU", "Daliyo RDP", "Bu Rini RDP", "Awan Pak Uus",
        "Pa Paryono RDP", "Novi RDP", "Pak Filbert RDP", "Pak Febert RDP", "Ibu Deki RDP",
        "Pak Waliyudinden", "Pa Joko RDP", "Bu Keke RDP", "Ipang RDP",
        "Pak Uus/ teh ita", "Foresh Hils B2", "Ibu Warti RDP", "Villa Bata Merah",
    ],
    "Harmoni (Mama Ola)": [
        "Umi Hani", "Ayman", "Bu Yuyu Rahayu", "Fahima", "Ibu Kartika", "Tante Kartika",
    ],
    "Aep Ciburial": [
        "Villa philanto RDP", "Villa Aito RDP", "Pa Aep", "Villa Forest Hill",
    ],
    "Bu Emi (UNISBA)": ["Bu Rini", "Bu Emi"],
    "SUKAVILLA": ["SUKAVILLA", "SUKAVILLA Pak hendi"],
}
def init_session():
    if "agents" not in st.session_state:
        st.session_state.agents = {k: list(v) for k, v in DEFAULT_AGENTS.items()}
# ─────────────────────────────────────────────
# STREAMLIT UI
# ─────────────────────────────────────────────
def main():
    st.set_page_config(page_title="20 Laundry - Invoice Generator", layout="wide")
    init_session()
    # ── Header ─────────────────────────────────────────────────────────────
    st.markdown("""
        <style>
        .app-header {
            background: linear-gradient(135deg, #1e3a5f 0%, #2e6da4 100%);
            color: white;
            padding: 1.4rem 1.8rem;
            border-radius: 10px;
            margin-bottom: 1.5rem;
        }
        .app-header h1 { margin: 0; font-size: 1.7rem; }
        .app-header p  { margin: 0.2rem 0 0; opacity: 0.8; font-size: 0.95rem; }
        .section-title { color: #1e3a5f; font-weight: 700; margin-top: 1.2rem; }
        div[data-testid="stExpander"] { border: 1px solid #d0dce8; border-radius: 8px; }
        </style>
        <div class="app-header">
            <h1>🧺 20 LAUNDRY — Invoice Generator</h1>
            <p>Generate agent invoices from ReBill CSV exports</p>
        </div>
    """, unsafe_allow_html=True)
    # ── Upload ──────────────────────────────────────────────────────────────
    st.markdown('<p class="section-title">1. Upload Data</p>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload file ReBill CSV", type=["csv"], label_visibility="collapsed")
    if not uploaded:
        st.info("Upload file ReBill CSV untuk memulai.", icon="📂")
        return
    # Load CSV
    try:
        df = load_rebill_df(uploaded, skiprows=5)
    except Exception as e:
        st.error(f"Gagal membaca CSV: {e}")
        return
    n_rows = len(df)
    date_range = ""
    if "Tanggal_dt" in df.columns and df["Tanggal_dt"].notna().any():
        d_min = df["Tanggal_dt"].min().strftime("%d %b %Y")
        d_max = df["Tanggal_dt"].max().strftime("%d %b %Y")
        date_range = f" | {d_min} - {d_max}"
    st.success(f"CSV berhasil dibaca: **{n_rows} transaksi**{date_range}", icon="✅")
    all_customers = sorted(
        df["Customer Name"].astype(str).str.strip().dropna().unique()
    )
    # ── Agent Setup ─────────────────────────────────────────────────────────
    st.markdown('<p class="section-title">2. Pengaturan Agent</p>', unsafe_allow_html=True)
    tab_select, tab_new = st.tabs(["Pilih Agent yang Ada", "Tambah Agent Baru"])
    # ── Tab: Select existing agent ──────────────────────────────────────────
    with tab_select:
        agent_name = st.selectbox(
            "Agent",
            list(st.session_state.agents.keys()),
            label_visibility="collapsed",
        )
        current_key = normalize_key(st.session_state.agents, agent_name)
        current_customers = sorted(
            {c.strip() for c in st.session_state.agents.get(current_key, []) if c.strip()},
            key=str.lower,
        )
        with st.expander(f"Edit customers untuk **{current_key}** ({len(current_customers)} terdaftar)"):
            options = sorted(set(all_customers) | set(current_customers), key=str.lower)
            edited = st.multiselect(
                "Pilih customers",
                options=options,
                default=current_customers,
                key="cust_multiselect",
            )
            extra_text = st.text_area(
                "Tambah customer lainnya (satu per baris atau pisahkan koma)",
                key="cust_extra",
                height=80,
            )
            if st.button("Simpan Perubahan", key="btn_save_customers"):
                combined = set(n.strip() for n in edited)
                combined.update(split_names(extra_text))
                combined.discard("")
                st.session_state.agents[current_key] = sorted(combined, key=str.lower)
                st.success(f"Tersimpan: {len(st.session_state.agents[current_key])} customers untuk '{current_key}'.")
    # ── Tab: Add new agent ──────────────────────────────────────────────────
    with tab_new:
        col_l, col_r = st.columns([1, 1])
        with col_l:
            new_agent_name = st.text_input("Nama Agent Baru", key="new_agent_name").strip()
        with col_r:
            picked = st.multiselect("Pilih dari daftar customers CSV", all_customers, key="new_agent_picked")
        extra_new = st.text_area(
            "Tambah customer lainnya",
            placeholder="Satu per baris atau pisahkan koma",
            key="new_agent_extra",
            height=80,
        )
        if st.button("Tambahkan Agent", key="btn_add_agent", type="primary"):
            if not new_agent_name:
                st.warning("Masukan nama agent terlebih dahulu.")
            else:
                existing_key = normalize_key(st.session_state.agents, new_agent_name)
                final_key = existing_key if existing_key in st.session_state.agents else new_agent_name
                base = set(st.session_state.agents.get(final_key, []))
                base.update(picked)
                base.update(split_names(extra_new))
                base.discard("")
                st.session_state.agents[final_key] = sorted(base, key=str.lower)
                agent_name = final_key
                st.success(f"Agent '{final_key}' berhasil ditambahkan dengan {len(st.session_state.agents[final_key])} customers.")
    # ── Invoice Parameters ───────────────────────────────────────────────────
    st.markdown('<p class="section-title">3. Parameter Invoice</p>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        # Default to the month actually present in the uploaded data, not
        # today's date - otherwise generating an invoice right after the
        # month rolls over (e.g. exporting August data on Sept 1st) silently
        # filters everything out and looks like a customer-matching bug.
        month_year = st.text_input("Bulan (YYYY-MM)", value=default_month_from_data(df))
    with col2:
        discount_pct = st.slider("Diskon Agent (%)", min_value=0, max_value=50, value=20, step=5)
        discount_rate = discount_pct / 100
    with col3:
        invoice_date = st.date_input("Tanggal Invoice", value=datetime.now().date())
        invoice_date_str = invoice_date.strftime("%d/%m/%Y")
    # ── Generate ─────────────────────────────────────────────────────────────
    st.markdown("")
    if st.button("Buat Invoice", type="primary", use_container_width=True):
        if not agent_name:
            st.warning("Pilih atau tambah agent terlebih dahulu.")
        else:
            try:
                invoice = build_invoice(df, st.session_state.agents, agent_name, month_year, discount_rate)
            except ValueError as e:
                st.error(str(e))
                return
            if not invoice["rows"]:
                st.warning(
                    f"Tidak ada transaksi untuk agent '{invoice['agent_key']}' "
                    f"pada periode {month_year}. "
                    "Pastikan nama customer sudah sesuai dengan data CSV, "
                    "dan periode di atas sesuai dengan rentang tanggal file CSV "
                    f"({date_range.strip(' |') if date_range else 'tidak diketahui'})."
                )
                return
            agent_key  = invoice["agent_key"]
            display_df = invoice_to_display_df(invoice)
            st.markdown(f'<p class="section-title">Invoice — {agent_key} | {month_year}</p>', unsafe_allow_html=True)
            st.markdown(f"**Tanggal Invoice:** {invoice_date_str} &nbsp;|&nbsp; **Diskon:** {discount_pct}%")
            # Summary metrics
            m1, m2, m3 = st.columns(3)
            m1.metric("Total Tagihan",  format_rp(invoice["totals"]["Price"]))
            m2.metric("Total Diskon",   format_rp(invoice["totals"]["Discount Agen (Rp)"]))
            m3.metric("Total Bayar",    format_rp(invoice["totals"]["Amount (Rp)"]))
            st.dataframe(display_df, use_container_width=True, hide_index=True)
            # ── PNG ────────────────────────────────────────────────────────
            img_title = (
                f"INVOICE  |  {agent_key}  |  Periode {month_year}\n"
                f"Tanggal Invoice: {invoice_date_str}    Diskon: {discount_pct}%"
            )
            img_buf = render_invoice_png(display_df, title=img_title)
            st.image(img_buf, caption="Preview Invoice (PNG)", use_container_width=True)
            # ── Excel ──────────────────────────────────────────────────────
            excel_buf = render_invoice_excel(display_df, agent_key, month_year, invoice_date_str)
            col_dl1, col_dl2 = st.columns(2)
            with col_dl1:
                st.download_button(
                    label="Download Invoice (PNG)",
                    data=img_buf,
                    file_name=f"Invoice_{agent_key.replace(' ', '_')}_{month_year}.png",
                    mime="image/png",
                    use_container_width=True,
                )
            with col_dl2:
                st.download_button(
                    label="Download Invoice (Excel)",
                    data=excel_buf,
                    file_name=f"Invoice_{agent_key.replace(' ', '_')}_{month_year}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                )
    # ── Sidebar: Agent directory ─────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### 📋 Daftar Agent")
        for ag, custs in st.session_state.agents.items():
            with st.expander(f"{ag} ({len(custs)} customers)"):
                for c in sorted(custs, key=str.lower):
                    st.markdown(f"- {c}")
        if st.button("Reset ke Default", key="btn_reset"):
            st.session_state.agents = {k: list(v) for k, v in DEFAULT_AGENTS.items()}
            st.success("Agent direset ke default.")
if __name__ == "__main__":
    main()
