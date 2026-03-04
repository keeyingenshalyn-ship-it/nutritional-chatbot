"""
NutriBot — Personalized AI Diet Planner
Built with Streamlit + Claude Opus 4.6

Demo premium access code: NUTRIBOT2024
"""

from pathlib import Path
from typing import Generator

import openai
import pandas as pd
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
APP_TITLE         = "NutriBot — Personalized Diet Planner"
DEMO_PREMIUM_CODE = "NUTRIBOT2024"   # Replace with real payment flow in production
FOOD_CSV          = Path(__file__).parent / "food.csv"

# Allergen → keywords to exclude from the food DB
ALLERGEN_KEYWORDS: dict[str, list[str]] = {
    "Gluten":     ["WHEAT", "BREAD", "PASTA", "CEREAL", "FLOUR", "BARLEY", "RYE", "CRACKERS"],
    "Dairy":      ["CHEESE", "BUTTER", "MILK", "CREAM", "YOGURT", "WHEY", "LACTOSE"],
    "Eggs":       ["EGG"],
    "Tree Nuts":  ["NUT", "ALMOND", "CASHEW", "WALNUT", "PECAN", "HAZELNUT", "PISTACHIO"],
    "Peanuts":    ["PEANUT"],
    "Soy":        ["SOY", "TOFU", "MISO", "TEMPEH"],
    "Shellfish":  ["SHRIMP", "CRAB", "LOBSTER", "CLAM", "OYSTER", "SCALLOP", "CRUSTACEAN"],
    "Fish":       ["FISH", "SALMON", "TUNA", "COD", "TILAPIA", "HALIBUT", "SARDINE"],
    "Sesame":     ["SESAME", "TAHINI"],
    "Corn":       ["CORN", "MAIZE"],
    "Nightshades":["TOMATO", "PEPPER", "EGGPLANT", "POTATO"],
}

ANIMAL_KEYWORDS = [
    "BEEF", "PORK", "CHICKEN", "TURKEY", "LAMB", "VEAL",
    "MEAT", "POULTRY", "SAUSAGE", "BACON", "HAM", "BISON",
]

# ─────────────────────────────────────────────────────────────────────────────
# Food CSV helpers
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_food_db() -> pd.DataFrame:
    """Load only the needed columns and pre-uppercase for fast filtering."""
    if not FOOD_CSV.exists():
        return pd.DataFrame()

    needed = [
        "Category", "Description",
        "Data.Kilocalories", "Data.Protein", "Data.Carbohydrate",
        "Data.Fat.Total Lipid", "Data.Fiber", "Data.Sugar Total",
    ]
    all_cols = pd.read_csv(FOOD_CSV, nrows=0).columns.tolist()
    use_cols = [c for c in needed if c in all_cols]

    df = pd.read_csv(FOOD_CSV, usecols=use_cols)
    df = df.rename(columns={
        "Description":          "Food",
        "Data.Kilocalories":    "Calories",
        "Data.Protein":         "Protein_g",
        "Data.Carbohydrate":    "Carbs_g",
        "Data.Fat.Total Lipid": "Fat_g",
        "Data.Fiber":           "Fiber_g",
        "Data.Sugar Total":     "Sugar_g",
    }).dropna(subset=["Calories"])

    # Pre-compute uppercase once — avoids repeated .str.upper() in filters
    df["_cat"]  = df["Category"].str.upper()
    df["_food"] = df["Food"].str.upper()
    return df


def _apply_keyword_mask(df: pd.DataFrame, keywords: list[str]) -> pd.DataFrame:
    """Remove rows whose Category or Food matches any keyword."""
    mask = pd.Series(True, index=df.index)
    for kw in keywords:
        mask &= ~(df["_cat"].str.contains(kw, na=False) |
                  df["_food"].str.contains(kw, na=False))
    return df[mask]


@st.cache_data(show_spinner=False)
def food_context_str(
    allergens: tuple[str, ...],
    preferences: tuple[str, ...],
) -> str:
    """Cached: filter the DB and return a compact CSV for the prompt."""
    df = load_food_db()
    if df.empty:
        return ""

    # Allergen exclusions
    exclude = []
    for a in allergens:
        exclude.extend(ALLERGEN_KEYWORDS.get(a, [a.upper()]))
    if exclude:
        df = _apply_keyword_mask(df, exclude)

    # Diet-style exclusions
    if "Vegan" in preferences:
        df = _apply_keyword_mask(
            df, ANIMAL_KEYWORDS + ["CHEESE", "BUTTER", "MILK", "CREAM", "YOGURT", "EGG"]
        )
    elif "Vegetarian" in preferences:
        df = _apply_keyword_mask(df, ANIMAL_KEYWORDS)

    if df.empty:
        return ""

    # Fast sampling: shuffle once, then take first 4 per category
    sample = (
        df.sample(frac=1, random_state=42)
          .groupby("Category")
          .head(4)
          .head(120)
          .reset_index(drop=True)
    )

    cols = ["Category", "Food", "Calories", "Protein_g", "Carbs_g", "Fat_g", "Fiber_g"]
    cols = [c for c in cols if c in sample.columns]
    return sample[cols].to_csv(index=False)

# ─────────────────────────────────────────────────────────────────────────────
# Trial helpers
# ─────────────────────────────────────────────────────────────────────────────

def trial_is_used() -> bool:
    """Return True if the free trial has been used in this session."""
    return st.session_state.get("trial_used", False)


def consume_trial() -> None:
    """Mark the free trial as used for this session."""
    st.session_state["trial_used"] = True


# ─────────────────────────────────────────────────────────────────────────────
# BMI helpers
# ─────────────────────────────────────────────────────────────────────────────
BMI_SCALE = [
    (18.5,        "Underweight", "#60A5FA"),
    (25.0,        "Normal",      "#34D399"),
    (30.0,        "Overweight",  "#FBBF24"),
    (float("inf"),"Obese",       "#F87171"),
]


def calc_bmi(weight_kg: float, height_cm: float) -> float:
    return round(weight_kg / (height_cm / 100) ** 2, 1)


def bmi_info(bmi: float) -> tuple[str, str]:
    for threshold, label, color in BMI_SCALE:
        if bmi < threshold:
            return label, color
    return "Obese", "#F87171"


# ─────────────────────────────────────────────────────────────────────────────
# Claude prompt
# ─────────────────────────────────────────────────────────────────────────────

def build_prompt(data: dict, include_recipes: bool) -> str:
    allergens   = ", ".join(data["allergens"])   or "None"
    preferences = ", ".join(data["preferences"]) or "No specific preferences"
    concerns    = data["health_concerns"].strip() or "None"

    recipe_instruction = (
        "For EVERY meal include a full step-by-step recipe: "
        "list all ingredients with exact quantities, then numbered cooking instructions."
        if include_recipes else
        "Provide a short (1–2 sentence) description of each meal. "
        "Do NOT include detailed recipes or ingredient lists."
    )

    food_ctx = food_context_str(
        tuple(data.get("allergens", [])),
        tuple(data.get("preferences", [])),
    )
    food_section = (
        f"\n═══════════════════════════════════════\n"
        f"VERIFIED FOOD NUTRITION DATA (per 100g)\n"
        f"═══════════════════════════════════════\n"
        f"Use these real values when selecting ingredients and calculating macros.\n"
        f"All figures are per 100 g of the food item.\n\n"
        f"{food_ctx}\n"
        if food_ctx else ""
    )

    return f"""You are a certified nutritionist and registered dietitian.
Create a detailed, personalized 7-day meal plan for the following individual.
{food_section}

═══════════════════════════════════════
PERSONAL PROFILE
═══════════════════════════════════════
• Age:    {data['age']} years
• Gender: {data['gender']}
• Weight: {data['weight_kg']:.1f} kg  ({data['weight_kg'] / 0.453592:.1f} lbs)
• Height: {data['height_cm']:.1f} cm  ({int(data['height_cm'] // 2.54 // 12)}' {int((data['height_cm'] / 2.54) % 12)}")
• BMI:    {data['bmi']} — {data['bmi_category']}

═══════════════════════════════════════
DIETARY REQUIREMENTS
═══════════════════════════════════════
• Allergens / foods to strictly avoid: {allergens}
• Dietary style / preferences:         {preferences}
• Health concerns / goals:             {concerns}
• Weekly grocery budget:               {data['budget']}

═══════════════════════════════════════
OUTPUT REQUIREMENTS
═══════════════════════════════════════
1. **Nutritional Assessment** — Brief intro, recommended daily calorie range, and
   macronutrient split (protein / carbs / fats) suited to this person's BMI and goals.

2. **7-Day Meal Plan** — For each day (Day 1 through Day 7):
   • Breakfast
   • Morning Snack
   • Lunch
   • Afternoon Snack
   • Dinner
   • Daily nutrition summary table:
     | Calories | Protein (g) | Carbs (g) | Fat (g) | Fiber (g) |

3. **Recipes** — {recipe_instruction}

4. **Weekly Grocery List** — Organized into sections:
   Produce | Proteins | Grains & Legumes | Dairy / Alternatives | Pantry Staples

5. **Meal Prep Tips** — 5 practical tips to save time and stay on budget.

6. **Health-Specific Notes** — Any important advice relating to the listed
   health concerns or goals (e.g., blood sugar management, sodium limits).

Format everything with clear Markdown headings and tables.
All recommendations must strictly avoid the listed allergens."""


# ─────────────────────────────────────────────────────────────────────────────
# Claude API — streaming
# ─────────────────────────────────────────────────────────────────────────────

def stream_plan(
    api_key: str,
    data: dict,
    include_recipes: bool,
) -> Generator[str, None, None]:
    """Stream the meal-plan text from ChatGPT 5."""
    client = openai.OpenAI(api_key=api_key)
    stream = client.chat.completions.create(
        model="gpt-5",
        max_completion_tokens=16000,
        stream=True,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a certified nutritionist and registered dietitian. "
                    "Provide practical, evidence-based dietary advice tailored to the individual."
                ),
            },
            {"role": "user", "content": build_prompt(data, include_recipes)},
        ],
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta is not None:
            yield delta


# ─────────────────────────────────────────────────────────────────────────────
# Session-state helpers
# ─────────────────────────────────────────────────────────────────────────────

def init_state() -> None:
    st.session_state.setdefault("step", 1)
    st.session_state.setdefault("user_data", {})
    st.session_state.setdefault("meal_plan", None)
    st.session_state.setdefault("premium", False)


def reset() -> None:
    for key in ("step", "user_data", "meal_plan", "premium"):
        st.session_state.pop(key, None)


# ─────────────────────────────────────────────────────────────────────────────
# Progress indicator
# ─────────────────────────────────────────────────────────────────────────────

def render_progress() -> None:
    step   = st.session_state.get("step", 1)
    labels = ["Personal Info", "BMI Analysis", "Preferences", "Meal Plan"]
    cols   = st.columns(4)
    for i, (col, lbl) in enumerate(zip(cols, labels), start=1):
        if i < step:
            col.success(f"✓ {lbl}")
        elif i == step:
            col.info(f"▶ {lbl}")
        else:
            col.markdown(
                f"<div style='color:#6B7280;text-align:center;padding:6px 0;'>"
                f"{i}. {lbl}</div>",
                unsafe_allow_html=True,
            )


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Personal information
# ─────────────────────────────────────────────────────────────────────────────

def render_step1() -> None:
    st.header("Step 1 — Personal Information")
    st.caption("Tell us about yourself so we can tailor your plan.")

    col1, col2 = st.columns(2)

    with col1:
        age    = st.number_input("Age", min_value=10, max_value=110, value=30, step=1)
        gender = st.selectbox("Gender", ["Male", "Female", "Non-binary / Prefer not to say"])
        unit   = st.radio("Unit system", ["Metric (kg / cm)", "Imperial (lbs / ft & in)"])

    with col2:
        if unit == "Metric (kg / cm)":
            weight_kg = st.number_input("Weight (kg)", min_value=20.0,  max_value=300.0, value=70.0,  step=0.5)
            height_cm = st.number_input("Height (cm)", min_value=100.0, max_value=250.0, value=170.0, step=0.5)
        else:
            lbs = st.number_input("Weight (lbs)", min_value=44.0, max_value=660.0, value=154.0, step=1.0)
            c1, c2 = st.columns(2)
            ft  = c1.number_input("Height (ft)", min_value=3, max_value=8, value=5, step=1)
            ins = c2.number_input("Height (in)", min_value=0, max_value=11, value=7, step=1)
            weight_kg = round(lbs * 0.453592, 2)
            height_cm = round((ft * 12 + ins) * 2.54, 2)

    st.markdown("")
    if st.button("Calculate BMI →", type="primary", use_container_width=True):
        st.session_state.user_data = {
            "age": age, "gender": gender,
            "weight_kg": weight_kg, "height_cm": height_cm,
        }
        st.session_state.step = 2
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — BMI analysis
# ─────────────────────────────────────────────────────────────────────────────

def render_step2() -> None:
    st.header("Step 2 — BMI Analysis")

    d     = st.session_state.user_data
    bmi   = calc_bmi(d["weight_kg"], d["height_cm"])
    label, color = bmi_info(bmi)
    d["bmi"]          = bmi
    d["bmi_category"] = label

    # Big BMI readout
    _, centre, _ = st.columns([1, 2, 1])
    with centre:
        st.markdown(
            f"""
            <div style="text-align:center;padding:28px 0;">
              <div style="font-size:15px;color:#9CA3AF;margin-bottom:4px;">Body Mass Index</div>
              <div style="font-size:80px;font-weight:900;color:{color};line-height:1;">{bmi}</div>
              <div style="font-size:22px;font-weight:700;color:{color};margin-top:4px;">{label}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # BMI colour scale
    scale_cols = st.columns(4)
    for (_, lbl, c), col in zip(BMI_SCALE, scale_cols):
        active = lbl.lower() == label.lower()
        border = "3px solid white" if active else "2px solid transparent"
        col.markdown(
            f"<div style='background:{c};color:white;padding:10px;border-radius:8px;"
            f"text-align:center;font-size:12px;font-weight:600;border:{border};'>{lbl}</div>",
            unsafe_allow_html=True,
        )

    st.divider()

    # Stats table
    ht_ft = int(d["height_cm"] // 2.54 // 12)
    ht_in = int((d["height_cm"] / 2.54) % 12)
    st.markdown(
        f"""
| Metric | Value |
|--------|-------|
| Weight | {d['weight_kg']:.1f} kg &nbsp;•&nbsp; {d['weight_kg']/0.453592:.1f} lbs |
| Height | {d['height_cm']:.1f} cm &nbsp;•&nbsp; {ht_ft}' {ht_in}" |
| BMI    | **{bmi}** — {label} |
        """
    )

    # BMI interpretation
    tips = {
        "Underweight": "Consider increasing calorie-dense whole foods. A high-protein, nutrient-rich plan may help.",
        "Normal":      "Great! We'll design a balanced plan to maintain your healthy weight.",
        "Overweight":  "A moderate calorie deficit with high fibre and lean protein can help you progress.",
        "Obese":       "We'll build a sustainable, medically mindful plan. Consult your doctor for additional support.",
    }
    st.info(f"**Note:** {tips.get(label, '')}")

    c1, c2 = st.columns(2)
    if c1.button("← Back", use_container_width=True):
        st.session_state.step = 1; st.rerun()
    if c2.button("Continue →", type="primary", use_container_width=True):
        st.session_state.step = 3; st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Diet preferences & health info
# ─────────────────────────────────────────────────────────────────────────────

def render_step3() -> None:
    st.header("Step 3 — Preferences & Health Info")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Allergens / Foods to Avoid")
        allergens = st.multiselect(
            "Select all that apply",
            ["Gluten", "Dairy", "Eggs", "Tree Nuts", "Peanuts", "Soy",
             "Shellfish", "Fish", "Sesame", "Corn", "Nightshades"],
            help="Your plan will strictly exclude these.",
        )

        st.subheader("Dietary Style")
        preferences = st.multiselect(
            "Select your diet type(s)",
            ["Vegan", "Vegetarian", "Pescatarian", "Keto / Low-Carb", "Paleo",
             "Mediterranean", "Halal", "Kosher", "High-Protein", "Low-Fat",
             "Diabetic-Friendly", "Heart-Healthy", "Anti-Inflammatory"],
        )

    with col2:
        st.subheader("Health Concerns & Goals")
        health_concerns = st.text_area(
            "Describe any conditions or objectives",
            placeholder=(
                "e.g., Type 2 diabetes, high blood pressure,\n"
                "weight-loss goal of 10 kg, building muscle, IBS..."
            ),
            height=130,
        )

        st.subheader("Weekly Grocery Budget")
        budget = st.select_slider(
            "Select your weekly budget",
            options=["Under $50", "$50–$100", "$100–$150", "$150–$200", "Over $200"],
            value="$50–$100",
        )

    c1, c2 = st.columns(2)
    if c1.button("← Back", use_container_width=True):
        st.session_state.step = 2; st.rerun()
    if c2.button("Generate My Meal Plan →", type="primary", use_container_width=True):
        st.session_state.user_data.update({
            "allergens":      allergens,
            "preferences":    preferences,
            "health_concerns": health_concerns,
            "budget":         budget,
        })
        st.session_state.step = 4
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Meal plan generation
# ─────────────────────────────────────────────────────────────────────────────

def render_step4(api_key: str) -> None:
    st.header("Step 4 — Your Personalized Meal Plan")

    trial_used = trial_is_used()
    premium    = st.session_state.get("premium", False)

    # ── Recipe access panel ──────────────────────────────────────────────────
    with st.container(border=True):
        st.markdown("#### Recipe Access")
        left, right = st.columns([3, 2])

        with left:
            if premium:
                st.success("⭐ **Premium active** — step-by-step recipes always included.")
                include_recipes = True

            elif not trial_used:
                st.success(
                    "🎁 **Free trial available!**  \n"
                    "You can unlock step-by-step recipes **once** at no cost."
                )
                include_recipes = st.checkbox(
                    "Include detailed step-by-step recipes *(uses your free trial)*",
                    value=False,
                )

            else:
                st.warning(
                    "🔒 **Free trial already used.**  \n"
                    "Upgrade to Premium to get step-by-step recipes on every plan."
                )
                include_recipes = False

        with right:
            st.markdown("**What's always included:**")
            st.markdown("- 7-day meal plan\n- Daily nutritional values\n- Weekly grocery list\n- Meal prep tips")
            if not premium and trial_used:
                st.markdown("---")
                st.markdown("**🔓 Unlock Premium**")
                code = st.text_input("Enter access code", placeholder="e.g. NUTRIBOT2024", key="premium_code")
                if st.button("Activate", key="activate_btn"):
                    if code.strip().upper() == DEMO_PREMIUM_CODE:
                        st.session_state.premium = True
                        st.success("Premium activated!")
                        st.rerun()
                    else:
                        st.error("Invalid code. Contact support@nutribot.app")

    st.divider()

    # ── Generate / display plan ──────────────────────────────────────────────
    if st.session_state.meal_plan is None:
        if st.button("🍽️  Generate My Meal Plan", type="primary", use_container_width=True):
            if not api_key:
                st.error("⚠️ Please enter your Anthropic API key in the sidebar.")
                return

            placeholder = st.empty()
            full_text   = ""
            error       = None

            with st.spinner("Crafting your personalized nutrition plan…"):
                try:
                    for chunk in stream_plan(api_key, st.session_state.user_data, include_recipes):
                        full_text += chunk
                        placeholder.markdown(full_text + "▌")
                    placeholder.markdown(full_text)
                    st.session_state.meal_plan = full_text

                    # Consume free trial if recipes were requested
                    if include_recipes and not trial_is_used() and not premium:
                        consume_trial()
                        st.toast("Free trial used — recipes unlocked for this plan!", icon="🎁")

                except openai.AuthenticationError:
                    error = "Invalid API key — please check the key you entered in the sidebar."
                except openai.RateLimitError:
                    error = "Rate limit reached — please wait a moment and try again."
                except openai.BadRequestError as e:
                    error = f"Bad request: {e}"
                except Exception as e:
                    error = f"Unexpected error: {e}"

            if error:
                st.error(error)
            else:
                st.rerun()

    else:
        # Render existing plan
        st.markdown(st.session_state.meal_plan)
        st.divider()

        c1, c2, c3 = st.columns(3)
        c1.download_button(
            "📥 Download Plan (.md)",
            data=st.session_state.meal_plan,
            file_name="meal_plan.md",
            mime="text/markdown",
            use_container_width=True,
        )
        if c2.button("🔄 Regenerate Plan", use_container_width=True):
            st.session_state.meal_plan = None
            st.rerun()
        if c3.button("🏠 Start Over", use_container_width=True):
            reset(); st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Main app
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="🥗",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Light style polish
    st.markdown(
        """
        <style>
        .stButton > button   { border-radius: 8px; font-weight: 600; }
        .main .block-container { max-width: 880px; padding-top: 1.5rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    init_state()
    load_food_db()          # Pre-warm CSV cache on every page load

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 🥗 NutriBot")
        st.caption("AI-powered nutrition advisor")
        st.divider()

        st.markdown("### 🔑 API Key")
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            placeholder="sk-...",
            help="Your key is used only for this session and is never stored.",
        )
        if api_key:
            st.caption("✅ Key entered")
        else:
            st.caption("Get a key at [platform.openai.com](https://platform.openai.com)")

        st.divider()

        step = st.session_state.get("step", 1)
        st.markdown("### Progress")
        st.progress((step - 1) / 4)
        st.caption(f"Step {step} of 4")

        st.divider()
        if trial_is_used():
            st.markdown("🔒 Free trial: **Used**")
        else:
            st.markdown("🎁 Free trial: **Available**")

        if step > 1:
            st.divider()
            if st.button("↺ Start Over", use_container_width=True):
                reset(); st.rerun()

    # ── Main content ─────────────────────────────────────────────────────────
    st.title("🥗 NutriBot")
    st.markdown("*Your AI-powered certified nutritionist — personalized diet plans in minutes.*")
    st.divider()

    render_progress()
    st.divider()

    step = st.session_state.get("step", 1)
    if   step == 1: render_step1()
    elif step == 2: render_step2()
    elif step == 3: render_step3()
    elif step == 4: render_step4(api_key)


if __name__ == "__main__":
    main()
