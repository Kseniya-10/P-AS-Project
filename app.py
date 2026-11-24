"""Dashboard and time-based validation for the Moscow property-price model."""
from pathlib import Path

import pandas as pd
import streamlit as st

from pipeline.config import load_config, resolve
from pipeline.model import evaluate_price_model
from pipeline.storage import CsvStore

st.set_page_config(
    page_title="Прогноз цены недвижимости",
    page_icon="🏠",
    layout="wide",
)

POLLUTANTS = {
    "pm2_5": "PM2.5, мкг/м³",
    "pm10": "PM10, мкг/м³",
    "nitrogen_dioxide": "NO₂, мкг/м³",
    "ozone": "O₃, мкг/м³",
}


def _read_mart():
    cfg = load_config()
    store = CsvStore(resolve(cfg, cfg["storage"]["directory"]))
    try:
        properties = store.read_table("mart.property_training")
        monthly = store.read_table("mart.monthly_price")
        air = store.read_table("mart.air_quality_daily")
    except (FileNotFoundError, ValueError) as exc:
        st.error(f"Не удалось прочитать CSV-витрины: {exc}")
        st.info("Сначала выполните `python -m pipeline run`.")
        st.stop()

    properties["date_posted"] = pd.to_datetime(properties["date_posted"], errors="coerce")
    numeric = [
        "lat", "lon", "price_per_sqm", "price_rub", "total_area",
        "pm10", "pm2_5", "nitrogen_dioxide", "ozone",
    ]
    for column in numeric:
        properties[column] = pd.to_numeric(properties[column], errors="coerce")
    monthly["month"] = pd.to_datetime(monthly["month"], errors="coerce")
    for column in ("offers", "mean_price_per_sqm", "median_price_per_sqm"):
        monthly[column] = pd.to_numeric(monthly[column], errors="coerce")
    air["date"] = pd.to_datetime(air["date"], errors="coerce")
    for column in POLLUTANTS:
        air[column] = pd.to_numeric(air[column], errors="coerce")
    return store.directory, properties, monthly, air


@st.cache_data(show_spinner="Обучаю модель и проверяю прогноз на поздних объявлениях…")
def _evaluate_model(frame: pd.DataFrame):
    return evaluate_price_model(frame)


@st.fragment(run_every="60s")
def dashboard():
    data_dir, properties, _, air = _read_mart()
    st.caption(f"CSV-витрины: `{Path(data_dir)}` · автообновление каждые 60 секунд")
    if properties.empty:
        st.warning("Витрина объявлений пуста. Проверьте источник данных и запустите конвейер.")
        return

    district_options = ["Все"] + sorted(properties["district"].dropna().astype(str).unique())
    okrug_options = ["Все"] + sorted(properties["okrug"].dropna().astype(str).unique())
    district_col, okrug_col = st.columns(2)
    district = district_col.selectbox("Район", district_options)
    okrug = okrug_col.selectbox("Округ", okrug_options)
    filtered = properties.copy()
    if district != "Все":
        filtered = filtered.loc[filtered["district"].eq(district)]
    if okrug != "Все":
        filtered = filtered.loc[filtered["okrug"].eq(okrug)]

    metrics = st.columns(4)
    metrics[0].metric("Объявлений", f"{len(filtered):,}".replace(",", " "))
    metrics[1].metric("Средняя цена, ₽/м²", f"{filtered['price_per_sqm'].mean():,.0f}")
    metrics[2].metric("Медианная цена, ₽/м²", f"{filtered['price_per_sqm'].median():,.0f}")
    metrics[3].metric("PM2.5, мкг/м³", f"{filtered['pm2_5'].mean():.1f}"
                      if filtered["pm2_5"].notna().any() else "нет данных")

    overview_tab, air_tab, forecast_tab = st.tabs(
        ["Рынок недвижимости", "Качество воздуха", "Прогноз цены за м²"]
    )
    with overview_tab:
        st.subheader("Динамика предложения")
        if not filtered.empty:
            monthly = filtered.assign(
                month=filtered["date_posted"].dt.to_period("M").dt.to_timestamp()
            ).groupby("month")["price_per_sqm"].agg(["mean", "median"])
            st.line_chart(
                monthly.rename(columns={"mean": "Средняя цена", "median": "Медианная цена"})
            )
        st.subheader("Расположение объявлений")
        map_frame = filtered[["lat", "lon"]].dropna()
        if not map_frame.empty:
            st.map(map_frame.rename(columns={"lat": "latitude", "lon": "longitude"}))
        st.subheader("Последние объявления")
        st.dataframe(
            filtered.sort_values("date_posted", ascending=False)[
                ["date_posted", "district", "okrug", "building_type", "total_area",
                 "rooms", "price_rub", "price_per_sqm", "pm2_5"]
            ].head(100),
            hide_index=True,
            width="stretch",
            column_config={
                "date_posted": st.column_config.DateColumn("Дата"),
                "price_rub": st.column_config.NumberColumn("Цена предложения, ₽", format="%.0f"),
                "price_per_sqm": st.column_config.NumberColumn("Цена за м², ₽", format="%.0f"),
                "pm2_5": st.column_config.NumberColumn("PM2.5, мкг/м³", format="%.1f"),
            },
        )

    with air_tab:
        st.subheader("Дневные показатели загрязнения воздуха")
        available = [column for column in POLLUTANTS if air[column].notna().any()]
        if not available:
            st.info("Данные о качестве воздуха пока не загружены. Запустите `python -m pipeline run`.")
        else:
            chosen = st.selectbox("Загрязнитель", available,
                                  format_func=lambda column: POLLUTANTS[column])
            series = air.dropna(subset=[chosen]).sort_values("date").set_index("date")[[chosen]]
            st.line_chart(series.rename(columns={chosen: POLLUTANTS[chosen]}))
            st.caption(
                "Качество воздуха загружается для центральной точки Москвы из исторического "
                "Open-Meteo Air Quality API; в признаки объявления попадает значение за "
                "предыдущий день, чтобы не использовать будущие измерения."
            )
            correlation = filtered[["price_per_sqm", "pm2_5", "pm10",
                                    "nitrogen_dioxide", "ozone"]].corr(numeric_only=True)
            st.subheader("Корреляции факторов и цены за м²")
            st.dataframe(correlation, width="stretch")

    with forecast_tab:
        st.subheader("Временная проверка модели")
        st.write(
            "Random Forest обучается на первых 80% периода и проверяется на оставшихся "
            "по времени объявлениях. В признаки включены характеристики жилья, локация, "
            "месяц публикации и загрязнители воздуха."
        )
        try:
            result = _evaluate_model(properties)
        except (ValueError, ImportError) as exc:
            st.warning(str(exc))
        else:
            score = st.columns(4)
            score[0].metric("MAE, ₽/м²", f"{result['mae']:,.0f}")
            score[1].metric("RMSE, ₽/м²", f"{result['rmse']:,.0f}")
            score[2].metric("R²", f"{result['r2']:.3f}")
            score[3].metric("Объектов test", f"{result['test_count']:,}".replace(",", " "))
            st.caption(
                f"Обучение: {result['train_count']:,} объявлений; тест — с "
                f"{result['cutoff']:%Y-%m-%d}. Целевой показатель — цена предложения, "
                "а не цена закрытой сделки."
            )
            st.dataframe(result["report"].head(200), hide_index=True, width="stretch")


st.title("Прогнозно-аналитическая система цены предложения недвижимости")
st.write(
    "Прогноз цены предложения за квадратный метр в Москве по характеристикам объекта "
    "и внешнему фактору — качеству воздуха."
)
dashboard()
