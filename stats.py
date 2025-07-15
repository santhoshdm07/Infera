import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.stats import power as sm_power
from statsmodels.stats.proportion import proportions_ztest
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import pingouin as pg
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
from sklearn.metrics import roc_curve, roc_auc_score

# Set page config
st.set_page_config(page_title="Infera - Statistical Analysis Platform", layout="wide")

# Initialize theme in session state
if 'theme' not in st.session_state:
    st.session_state.theme = "light"

# Define vibrant color palettes
COLOR_PALETTES = {
    "Vibrant": ["#FF6B6B", "#4ECDC4", "#45B7D1", "#FFA07A", "#98D8C8", "#F06292", "#7986CB", "#9575CD", "#64B5F6",
                "#4DB6AC"],
    "Rainbow": ["#FF0000", "#FF7F00", "#FFFF00", "#00FF00", "#0000FF", "#4B0082", "#9400D3"],
    "Pastel": ["#FFD1DC", "#FFECB8", "#B5EAD7", "#C7CEEA", "#E2F0CB", "#FFDAC1", "#B5EAD7"],
    "Cool": ["#003F5C", "#2F4B7C", "#665191", "#A05195", "#D45087", "#F95D6A", "#FF7C43", "#FFA600"],
    "Warm": ["#FF5E78", "#FF8A5E", "#FFAA5E", "#FFC45E", "#FFDC5E", "#FFEE5E", "#FFF75E"],
    "Ocean": ["#00A8E8", "#0077B6", "#0096C7", "#00B4D8", "#48CAE4", "#90E0EF", "#ADE8F4"],
    "Neon": ["#FF00FF", "#00FFFF", "#FF00FF", "#00FF00", "#FFFF00", "#FF0000", "#0000FF"]
}


# Function to get colors based on selection
def get_colors(palette_name, n_colors):
    palette = COLOR_PALETTES.get(palette_name, COLOR_PALETTES["Vibrant"])
    return [palette[i % len(palette)] for i in range(n_colors)]


with st.sidebar:
    st.markdown("""
        <div style="padding: 7px 12px; border-radius: 1px; background-color: #f0f2f6;">
            <span style="font-weight: 600; font-size: 13px;">
                📘 Quick Reference – 
                <a href="https://www.stratascratch.com/blog/a-comprehensive-statistics-cheat-sheet-for-data-science-interviews/" 
                   target="_blank" 
                   style="text-decoration: none; color: #1a73e8;">
                    Statistics Cheat Sheet
                </a>
            </span>
        </div>
    """, unsafe_allow_html=True)





def sidebar_links():
    st.sidebar.markdown("""
    <style>
    .sidebar-bottom-links {
        display: flex;
        flex-direction: column;
        height: 32vh;
        justify-content: flex-end;
        padding-bottom: 20px;
    }
    .icon-row {
        display: flex;
        gap: 14px;
        align-items: center;
        padding-left: 6px;
    }
    .icon-row img {
        width: 20px;
        height: 20px;
        opacity: 0.85;
        transition: transform 0.2s ease;
    }
    .icon-row img:hover {
        transform: scale(1.2);
        opacity: 1;
    }
    </style>

    <div class="sidebar-bottom-links">
        <div style="font-size: 17px; color: black; margin-left: 6px;">Connect with me..</div>
        <div class="icon-row">
            <a href="https://www.linkedin.com/in/santhosh-dm-analyst/" target="_blank" title="LinkedIn">
                <img src="https://cdn.jsdelivr.net/npm/simple-icons@v5/icons/linkedin.svg">
            </a>
            <a href="https://github.com/santhoshdm07/-santhoshdm07" target="_blank" title="GitHub">
                <img src="https://cdn.jsdelivr.net/npm/simple-icons@v5/icons/github.svg">
            </a>
            <a href="https://www.kaggle.com/santhosh77" target="_blank" title="Kaggle">
                <img src="https://cdn.jsdelivr.net/npm/simple-icons@v5/icons/kaggle.svg">
            </a>
            <a href="mailto:santhosh18.dm@gmail.com" title="Gmail">
                <img src="https://cdn.jsdelivr.net/npm/simple-icons@v5/icons/gmail.svg">
            </a>
        </div>
    </div>
    """, unsafe_allow_html=True)




if 'theme' not in st.session_state:
    st.session_state.theme = "light"


def apply_theme():
    theme = st.session_state.theme
    background = '#ffffff' if theme == 'light' else '#121212'
    text_color = '#2c3e50' if theme == 'light' else '#e0e0e0'
    input_bg = '#f0f2f6' if theme == 'light' else '#1e1e1e'
    border_color = '#ccc' if theme == 'light' else '#333'

    st.markdown(f"""
    <style>
        /* Main background and text */
        .stApp {{
            background-color: {background};
            color: {text_color};
        }}
        html, body, div, span, h1, h2, h3, h4, h5, h6, p {{
            color: {text_color};
        }}

        /* Sidebar background and text */
        section[data-testid="stSidebar"] {{
            background-color: {input_bg};
            color: {text_color};
        }}
        section[data-testid="stSidebar"] * {{
            color: {text_color};
        }}

        /* Inputs, textareas, selects */
        input, textarea, select {{
            background-color: {input_bg};
            color: {text_color};
            border: 1px solid {border_color};
        }}

        /* Dropdown and selectbox (deep targeting) */
        div[data-baseweb="select"] > div {{
            background-color: {input_bg} !important;
            color: {text_color} !important;
        }}
        div[data-baseweb="select"] * {{
            color: {text_color} !important;
        }}

        /* Buttons */
        .stButton > button {{
            background-color: {'#4CAF50' if theme == 'light' else '#2e7d32'};
            color: white;
            border: none;
        }}
        .stDownloadButton > button {{
            background-color: {'#2196F3' if theme == 'light' else '#1565c0'};
            color: white;
        }}

        /* DataFrame and charts */
        .stDataFrame, .stTable {{
            background-color: {input_bg} !important;
            color: {text_color} !important;
        }}
    </style>
    """, unsafe_allow_html=True)



def toggle_theme():
    st.session_state.theme = "dark" if st.session_state.theme == "light" else "light"


apply_theme()

st.button(
    f"{'🌙' if st.session_state.theme == 'light' else '☀️'} Toggle Theme",
    on_click=toggle_theme
)

# st.write(f"Current theme: **{st.session_state.theme.capitalize()}**")



# Main app
def main():
    # Header Section with improved layout and styling
    st.markdown("""
        <style>
            .header-container {
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 0.5rem 0 1rem 0;
            }
            .app-title {
                font-size: 2.5rem;
                font-weight: 700;
                margin-bottom: 0.1rem;
                color: var(--text);
            }
            .app-subtitle {
                font-size: 1.3rem; /* Increased size */
                font-weight: 500;
                color: var(--text-secondary);
            }
            hr {
                border: none;
                border-top: 1px solid var(--border);
                margin-top: 1rem;
                margin-bottom: 1.5rem;
            }
        </style>

        <div class="header-container">
            <div>
                <div class="app-title">Infera</div>
                <div class="app-subtitle">Comprehensive Statistical Analysis Platform</div>
            </div>
            <!-- Optional: insert logo or theme toggle button -->
        </div>
        <hr>
    """, unsafe_allow_html=True)


    # Introduction Text
    st.markdown("""
    **This platform provides a complete suite of statistical analysis tools, including:**
    - Descriptive statistics
    - Hypothesis testing
    - Correlation analysis
    - Regression analysis
    - ANOVA
    - Non-parametric tests
    - Probability Distribution Simulations
    - And much more!
    """)

    # Initialize session state for data and color palette
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'color_palette' not in st.session_state:
        st.session_state.color_palette = "Vibrant"

    # Sidebar for data upload and options
    with st.sidebar:
        st.header(" 📥 Data Input")
        uploaded_file = st.file_uploader("Upload your dataset (CSV or Excel)", type=["csv", "xlsx"])

        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    st.session_state.df = pd.read_csv(uploaded_file)
                else:
                    st.session_state.df = pd.read_excel(uploaded_file)
                st.success("Data loaded successfully!")
            except Exception as e:
                st.error(f"Error loading file: {e}")

        # Visualization Options
        st.header("Visualization Options")

        st.markdown("""
            <style>
                .tight-label {
                    font-size: 16px;
                    font-weight: 600;
                    margin-bottom: 0px;
                    margin-top: -10px;
                }
                .block-container .element-container:has(.tight-label) + div {
                    margin-top: -10px;
                }
            </style>
            <div class='tight-label'>
                🎨 Select Color Palette
            </div>
        """, unsafe_allow_html=True)

        st.session_state.color_palette = st.selectbox(
            "",  # Empty label since we're using custom HTML label
            list(COLOR_PALETTES.keys()),
            index=list(COLOR_PALETTES.keys()).index(st.session_state.color_palette)
        )

        # Analysis Options
        st.header("Analysis Options")

        st.markdown("""
            <div class='tight-label'>
                🔍 Select Analysis Type
            </div>
        """, unsafe_allow_html=True)

        analysis_type = st.selectbox(
            "",
            [
                "Descriptive Statistics",
                "Normality Tests",
                "T-tests",
                "ANOVA",
                "Correlation Analysis",
                "Chi-square Tests",
                "Non-parametric Tests",
                "Regression Analysis",
                "Proportion Tests",
                "Probability Distributions",
                "Central Limit Theorem"
            ]
        )

    st.markdown("---")
    sidebar_links()

    # Main content area
    if st.session_state.df is not None:
        st.subheader("Data Preview")
        st.dataframe(st.session_state.df.head())

        # Get numeric and categorical columns
        numeric_cols = st.session_state.df.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = st.session_state.df.select_dtypes(exclude=np.number).columns.tolist()

        # Perform selected analysis
        if analysis_type == "Descriptive Statistics":
            descriptive_statistics(numeric_cols)
        elif analysis_type == "Normality Tests":
            normality_tests(numeric_cols)
        elif analysis_type == "T-tests":
            t_tests(numeric_cols)
        elif analysis_type == "ANOVA":
            anova_tests(numeric_cols, categorical_cols)
        elif analysis_type == "Correlation Analysis":
            correlation_analysis(numeric_cols)
        elif analysis_type == "Chi-square Tests":
            chi_square_tests(categorical_cols)
        elif analysis_type == "Non-parametric Tests":
            nonparametric_tests(numeric_cols, categorical_cols)
        elif analysis_type == "Regression Analysis":
            regression_analysis(numeric_cols)
        elif analysis_type == "Power Analysis":
            power_analysis()
        elif analysis_type == "Proportion Tests":
            proportion_tests()
        elif analysis_type == "Probability Distributions":
            probability_distributions()
        elif analysis_type == "Central Limit Theorem":
            central_limit_theorem()
    else:
        st.info("Please upload a dataset to begin analysis.")


# Probability Distribution Simulations
def probability_distributions():
    st.header("Probability Distribution Simulations")

    st.markdown("""
    <div class="test-info">
    <h4>About Probability Distributions</h4>
    <p>Probability distributions describe how probabilities are distributed over the values of a random variable. They are fundamental to statistical modeling and hypothesis testing.</p>

    <p><strong>Key Concepts:</strong></p>
    <ul>
        <li><strong>PMF (Probability Mass Function):</strong> For discrete distributions, gives probability of each possible value</li>
        <li><strong>PDF (Probability Density Function):</strong> For continuous distributions, describes relative likelihood</li>
        <li><strong>CDF (Cumulative Distribution Function):</strong> Probability that X will take a value ≤ x</li>
        <li><strong>Expected Value:</strong> Long-run average value of repetitions</li>
        <li><strong>Variance:</strong> Measure of dispersion from the mean</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="example-box">
    <h4>Practical Applications:</h4>
    <ul>
        <li><strong>Bernoulli:</strong> Modeling single yes/no outcomes (coin flips, success/failure)</li>
        <li><strong>Binomial:</strong> Counting successes in fixed trials (defect rates, survey responses)</li>
        <li><strong>Poisson:</strong> Modeling rare events over time (customer arrivals, system failures)</li>
        <li><strong>Geometric:</strong> Time until first success (equipment lifespan, trial-and-error processes)</li>
    </ul>

    <h4>Example Scenarios:</h4>
    <ul>
        <li>A manufacturing plant uses Binomial distribution to model defective items in batches</li>
        <li>Call centers use Poisson distribution to predict incoming call volumes</li>
        <li>Quality control uses Geometric distribution to estimate inspection intervals</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    dist_type = st.selectbox(
        "**Select distribution**",
        ["Bernoulli", "Binomial", "Poisson", "Geometric"]
    )

    colors = get_colors(st.session_state.color_palette, 2)

    if dist_type == "Bernoulli":
        st.subheader("Bernoulli Distribution")
        st.markdown("""
        <div class="note">
        <h4>Notes:</h4>
        <ul>
            <li>Models a single trial with two possible outcomes (success=1, failure=0)</li>
            <li>Parameter p = probability of success</li>
            <li>Mean = p, Variance = p(1-p)</li>
            <li>Foundation for many other distributions</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            p = st.slider("Probability of success (p)", 0.0, 1.0, 0.5, 0.01)
            n_trials = st.slider("Number of trials", 1, 1000, 100)
        with col2:
            show_theoretical = st.checkbox("Show theoretical distribution", True)
            show_cdf = st.checkbox("Show cumulative distribution", False)

        # Simulate Bernoulli trials
        trials = np.random.binomial(1, p, n_trials)
        counts = pd.Series(trials).value_counts().sort_index()

        # Create figure
        fig = go.Figure()

        # Empirical distribution
        fig.add_trace(go.Bar(
            x=counts.index,
            y=counts / n_trials,
            name='Empirical',
            marker_color=colors[0],
            opacity=0.7
        ))

        # Theoretical distribution
        if show_theoretical:
            theoretical = pd.Series({
                0: (1 - p),
                1: p
            })
            fig.add_trace(go.Scatter(
                x=theoretical.index,
                y=theoretical,
                mode='markers',
                name='Theoretical',
                marker=dict(color=colors[1], size=12)
            ))

        # CDF if requested
        if show_cdf:
            empirical_cdf = np.cumsum(counts / n_trials)
            fig.add_trace(go.Scatter(
                x=counts.index,
                y=empirical_cdf,
                mode='lines+markers',
                name='Empirical CDF',
                line=dict(color=colors[0], dash='dot')
            ))

            if show_theoretical:
                theoretical_cdf = np.cumsum(theoretical)
                fig.add_trace(go.Scatter(
                    x=theoretical.index,
                    y=theoretical_cdf,
                    mode='lines+markers',
                    name='Theoretical CDF',
                    line=dict(color=colors[1], dash='dot')
                ))

        fig.update_layout(
            title=f"Bernoulli Distribution (p={p})",
            xaxis_title="Outcome",
            yaxis_title="Probability",
            xaxis=dict(tickvals=[0, 1], ticktext=["Failure (0)", "Success (1)"]),
            barmode='group'
        )
        st.plotly_chart(fig, use_container_width=True)

        # Statistics display
        st.subheader("Distribution Statistics")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Empirical Mean", f"{trials.mean():.4f}")
            st.metric("Theoretical Mean", f"{p:.4f}")
        with col2:
            st.metric("Empirical Variance", f"{trials.var():.4f}")
            st.metric("Theoretical Variance", f"{p * (1 - p):.4f}")

    elif dist_type == "Binomial":
        st.subheader("Binomial Distribution")
        st.markdown("""
        <div class="note">
        <h4>Notes:</h4>
        <ul>
            <li>Models number of successes in n independent Bernoulli trials</li>
            <li>Parameters: n = number of trials, p = success probability</li>
            <li>Mean = np, Variance = np(1-p)</li>
            <li>Approaches Normal distribution when n is large (Central Limit Theorem)</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            p = st.slider("Probability of success (p)", 0.0, 1.0, 0.5, 0.01)
            n = st.slider("Number of trials per experiment (n)", 1, 100, 10)
        with col2:
            n_experiments = st.slider("Number of experiments", 1, 5000, 1000)
            show_normal = st.checkbox("Show normal approximation", True)

        # Simulate Binomial experiments
        experiments = np.random.binomial(n, p, n_experiments)

        # Create figure
        fig = go.Figure()

        # Histogram of results
        fig.add_trace(go.Histogram(
            x=experiments,
            name='Empirical',
            marker_color=colors[0],
            opacity=0.7,
            histnorm='probability'
        ))

        # Theoretical PMF
        x = np.arange(0, n + 1)
        pmf = stats.binom.pmf(x, n, p)
        fig.add_trace(go.Scatter(
            x=x,
            y=pmf,
            mode='markers',
            name='Theoretical PMF',
            marker=dict(color=colors[1], size=8)
        ))

        # Normal approximation if requested
        if show_normal and n * p > 5 and n * (1 - p) > 5:  # Rule of thumb for normal approx
            mu = n * p
            sigma = np.sqrt(n * p * (1 - p))
            x_norm = np.linspace(0, n, 100)
            norm_pdf = stats.norm.pdf(x_norm, mu, sigma)
            fig.add_trace(go.Scatter(
                x=x_norm,
                y=norm_pdf,
                mode='lines',
                name='Normal Approx',
                line=dict(color='green', dash='dash')
            ))

        fig.update_layout(
            title=f"Binomial Distribution (n={n}, p={p})",
            xaxis_title="Number of Successes",
            yaxis_title="Probability",
            bargap=0.1
        )
        st.plotly_chart(fig, use_container_width=True)

        # Statistics display
        st.subheader("Distribution Statistics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Empirical Mean", f"{experiments.mean():.4f}")
            st.metric("Theoretical Mean", f"{n * p:.4f}")
        with col2:
            st.metric("Empirical Variance", f"{experiments.var():.4f}")
            st.metric("Theoretical Variance", f"{n * p * (1 - p):.4f}")
        with col3:
            st.metric("Empirical Skewness", f"{stats.skew(experiments):.4f}")
            st.metric("Theoretical Skewness", f"{(1 - 2 * p) / np.sqrt(n * p * (1 - p)):.4f}")


    elif dist_type == "Poisson":
        st.subheader("Poisson Distribution")
        st.markdown("""
        <div class="note">
        <h4>Notes:</h4>
        <ul>
            <li>Models rare events occurring in fixed interval of time/space</li>
            <li>Parameter λ = average rate of occurrence</li>
            <li>Mean = λ, Variance = λ</li>
            <li>Used in queueing theory, reliability analysis, and risk assessment</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            lam = st.slider("Rate parameter (λ)", 0.1, 20.0, 5.0, 0.1)
            n_experiments = st.slider("Number of experiments", 1, 5000, 1000)
        with col2:
            show_diff = st.checkbox("Show difference from theoretical", False)
            show_poisson_process = st.checkbox("Show Poisson process simulation", False)

        # Simulate Poisson experiments
        experiments = np.random.poisson(lam, n_experiments)

        # Create main figure
        fig = go.Figure()

        # Histogram of results
        fig.add_trace(go.Histogram(
            x=experiments,
            name='Empirical',
            marker_color=colors[0],
            opacity=0.7,
            histnorm='probability'
        ))

        # Theoretical PMF
        x = np.arange(0, int(lam * 3) + 1)
        pmf = stats.poisson.pmf(x, lam)
        fig.add_trace(go.Scatter(
            x=x,
            y=pmf,
            mode='markers',
            name='Theoretical PMF',
            marker=dict(color=colors[1], size=8)
        ))

        fig.update_layout(
            title=f"Poisson Distribution (λ={lam})",
            xaxis_title="Number of Events",
            yaxis_title="Probability",
            bargap=0.1
        )
        st.plotly_chart(fig, use_container_width=True)

        # Difference plot if requested
        if show_diff:
            # Calculate empirical PMF first
            empirical_pmf = np.array([np.mean(experiments == k) for k in x])
            pmf = stats.poisson.pmf(x, lam)  # Recalculate to be safe
            diff = empirical_pmf - pmf

            fig_diff = go.Figure()
            fig_diff.add_trace(go.Bar(
                x=x,
                y=diff,
                name='Difference (Empirical - Theoretical)',
                marker_color='orange'
            ))
            fig_diff.update_layout(
                title="Difference Between Empirical and Theoretical PMF",
                xaxis_title="Number of Events",
                yaxis_title="Probability Difference",
                bargap=0.1
            )
            st.plotly_chart(fig_diff, use_container_width=True)

        # Poisson process simulation if requested
        if show_poisson_process:
            st.subheader("Poisson Process Simulation")
        time_period = st.slider("Time period (T)", 1.0, 10.0, 5.0, 0.1)

        # Generate event times
        event_times = np.cumsum(np.random.exponential(1 / lam, size=int(lam * time_period * 2)))
        event_times = event_times[event_times <= time_period]

        # Create timeline plot
        fig_process = go.Figure()
        for i, t in enumerate(event_times):
            fig_process.add_trace(go.Scatter(
                x=[t, t],
                y=[0, 1],
                mode='lines',
                line=dict(color=colors[i % len(colors)], width=2),
                showlegend=False
            ))

        fig_process.update_layout(
            title=f"Poisson Process Timeline (λ={lam}, T={time_period})",
            xaxis_title="Time",
            yaxis=dict(showticklabels=False),
            height=200
        )
        st.plotly_chart(fig_process, use_container_width=True)

        st.write(f"Number of events in {time_period} time units: {len(event_times)}")
        st.write(f"Average time between events: {np.mean(np.diff(event_times)):.4f} (expected: {1 / lam:.4f})")

        # Statistics display
        st.subheader("Distribution Statistics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Empirical Mean", f"{experiments.mean():.4f}")
            st.metric("Theoretical Mean", f"{lam:.4f}")
        with col2:
            st.metric("Empirical Variance", f"{experiments.var():.4f}")
            st.metric("Theoretical Variance", f"{lam:.4f}")
        with col3:
            st.metric("Empirical Skewness", f"{stats.skew(experiments):.4f}")
            st.metric("Theoretical Skewness", f"{1 / np.sqrt(lam):.4f}")


    elif dist_type == "Geometric":
        st.subheader("Geometric Distribution")
        st.markdown("""
        <div class="note">
        <h4>Notes:</h4>
        <ul>
            <li>Models number of trials until first success</li>
            <li>Parameter p = probability of success on each trial</li>
            <li>Mean = 1/p, Variance = (1-p)/p²</li>
            <li>Memoryless property: P(X>m+n|X>m) = P(X>n)</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            p = st.slider("Probability of success (p)", 0.01, 1.0, 0.5, 0.01)
            n_experiments = st.slider("Number of experiments", 1, 5000, 1000)
        with col2:
            show_memoryless = st.checkbox("Demonstrate memoryless property", True)
            max_trials = st.slider("Maximum trials to show", 1, 50, 20)

        # Simulate Geometric experiments
        experiments = np.random.geometric(p, n_experiments)

        # Create figure
        fig = go.Figure()

        # Histogram of results (limited to max_trials)
        valid_experiments = experiments[experiments <= max_trials]
        fig.add_trace(go.Histogram(
            x=valid_experiments,
            name='Empirical',
            marker_color=colors[0],
            opacity=0.7,
            histnorm='probability'
        ))

        # Theoretical PMF
        x = np.arange(1, max_trials + 1)
        pmf = stats.geom.pmf(x, p)
        fig.add_trace(go.Scatter(
            x=x,
            y=pmf,
            mode='markers',
            name='Theoretical PMF',
            marker=dict(color=colors[1], size=8)
        ))

        fig.update_layout(
            title=f"Geometric Distribution (p={p})",
            xaxis_title="Number of Trials Until First Success",
            yaxis_title="Probability",
            bargap=0.1
        )
        st.plotly_chart(fig, use_container_width=True)

        # Memoryless property demonstration
        if show_memoryless:
            st.subheader("Memoryless Property Demonstration")
        m = st.slider("Given already waited m trials without success", 1, 10, 2)

        # Filter experiments where X > m
        conditioned = experiments[experiments > m]
        additional_wait = conditioned - m

        # Compare distributions
        fig_mem = make_subplots(rows=1, cols=2, subplot_titles=[
            f"Original P(X > {m})",
            f"Conditional P(X > {m} + n | X > {m})"
        ])

        # Original distribution beyond m
        fig_mem.add_trace(go.Histogram(
            x=experiments[experiments > m],
            name='Original',
            marker_color=colors[0],
            opacity=0.7,
            histnorm='probability'
        ), row=1, col=1)

        # Conditional distribution
        fig_mem.add_trace(go.Histogram(
            x=additional_wait,
            name='Conditional',
            marker_color=colors[1],
            opacity=0.7,
            histnorm='probability'
        ), row=1, col=2)

        fig_mem.update_layout(
            title=f"Memoryless Property: P(X>{m}+n|X>{m}) = P(X>n)",
            showlegend=False,
            bargap=0.1
        )
        st.plotly_chart(fig_mem, use_container_width=True)

        # Compare probabilities
        st.write(f"P(X > {m}) = {len(experiments[experiments > m]) / len(experiments):.4f}")
        st.write(f"P(X > {m + 2}) = {len(experiments[experiments > m + 2]) / len(experiments):.4f}")
        st.write(f"P(X > 2) = {len(experiments[experiments > 2]) / len(experiments):.4f}")
        st.write(
            f"P(X > {m + 2} | X > {m}) = {len(experiments[experiments > m + 2]) / len(experiments[experiments > m]):.4f} ≈ P(X > 2)")

        # Statistics display
        st.subheader("Distribution Statistics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Empirical Mean", f"{experiments.mean():.4f}")
            st.metric("Theoretical Mean", f"{1 / p:.4f}")
        with col2:
            st.metric("Empirical Variance", f"{experiments.var():.4f}")
            st.metric("Theoretical Variance", f"{(1 - p) / (p ** 2):.4f}")
        with col3:
            st.metric("Empirical P(X ≤ 5)", f"{np.mean(experiments <= 5):.4f}")
            st.metric("Theoretical P(X ≤ 5)", f"{1 - (1 - p) ** 5:.4f}")

        # Central Limit Theorem Demonstration


def central_limit_theorem():
    st.header("Central Limit Theorem Demonstration")

    st.markdown("""
    <div class="test-info">
    <h4>About the Central Limit Theorem (CLT)</h4>
    <p>The CLT states that the sampling distribution of the mean of any independent, random variable 
    will be normal or nearly normal, if the sample size is large enough.</p>

    <p><strong>Key Implications:</strong></p>
    <ul>
        <li>Works regardless of the population distribution shape</li>
        <li>Sample means will be approximately normally distributed</li>
        <li>Mean of sampling distribution = Population mean (μ)</li>
        <li>Standard error = σ/√n (σ = population standard deviation)</li>
    </ul>

    <p><strong>Rule of Thumb:</strong> n ≥ 30 for reasonable approximation</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="example-box">
    <h4>Practical Applications:</h4>
    <ul>
        <li>Justifies use of normal distribution in many statistical tests</li>
        <li>Foundation for confidence intervals and hypothesis testing</li>
        <li>Quality control (control charts)</li>
        <li>Risk management in finance</li>
    </ul>

    <h4>Example Scenarios:</h4>
    <ul>
        <li>A factory uses CLT to monitor average product weights, even when individual weights are skewed</li>
        <li>Pollsters use CLT to estimate sampling error in election polls</li>
        <li>Financial analysts assume normal distribution for portfolio returns over time</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        pop_dist = st.selectbox(
            "**Population distribution**",
            ["Uniform", "Exponential", "Poisson", "Binomial", "Lognormal", "Weibull"],
            index=1
        )
        sample_size = st.slider("Sample size (n)", 1, 100, 30)
    with col2:
        n_samples = st.slider("Number of samples", 10, 10000, 1000)
        show_population = st.checkbox("Show population distribution", True)

    # Generate population data based on selected distribution
    if pop_dist == "Uniform":
        pop_data = np.random.uniform(0, 100, 100000)
        pop_name = "Uniform(0,100)"
        true_mean = 50
        true_std = np.sqrt((100 ** 2) / 12)  # (b-a)²/12
    elif pop_dist == "Exponential":
        pop_data = np.random.exponential(1, 100000)
        pop_name = "Exponential(1)"
        true_mean = 1
        true_std = 1
    elif pop_dist == "Poisson":
        lam = 5
        pop_data = np.random.poisson(lam, 100000)
        pop_name = f"Poisson({lam})"
        true_mean = lam
        true_std = np.sqrt(lam)
    elif pop_dist == "Binomial":
        n, p = 10, 0.5
        pop_data = np.random.binomial(n, p, 100000)
        pop_name = f"Binomial({n},{p})"
        true_mean = n * p
        true_std = np.sqrt(n * p * (1 - p))
    elif pop_dist == "Lognormal":
        pop_data = np.random.lognormal(0, 1, 100000)
        pop_name = "Lognormal(0,1)"
        true_mean = np.exp(0 + 0.5)  # exp(μ + σ²/2)
        true_std = np.sqrt((np.exp(1) - 1) * np.exp(1))  # sqrt((e^σ²-1)e^{2μ+σ²})
    elif pop_dist == "Weibull":
        a = 1.5
        pop_data = np.random.weibull(a, 100000)
        pop_name = f"Weibull({a})"
        true_mean = stats.gamma(1 + 1 / a)  # Γ(1 + 1/a)
        true_std = np.sqrt(stats.gamma(1 + 2 / a) - (stats.gamma(1 + 1 / a)) ** 2)

    # Sample means
    sample_means = [np.mean(np.random.choice(pop_data, sample_size)) for _ in range(n_samples)]
    colors = ['#1f77b4', '#ff7f0e']
    # Create subplots
    fig = make_subplots(
        rows=1,
        cols=2 if show_population else 1,
        subplot_titles=[
            "Sampling Distribution of Mean",
            "Population Distribution"
        ] if show_population else ["Sampling Distribution of Mean"]
    )

    # Plot sampling distribution
    fig.add_trace(
        go.Histogram(
            x=sample_means,
            name='Sample Means',
            marker_color=colors[0],
            opacity=0.7,
            histnorm='probability density'
        ),
        row=1, col=1
    )

    # Add normal curve fit
    x = np.linspace(min(sample_means), max(sample_means), 100)
    pdf = stats.norm.pdf(x, np.mean(sample_means), np.std(sample_means))
    fig.add_trace(
        go.Scatter(
            x=x,
            y=pdf,
            mode='lines',
            name='Normal Fit',
            line=dict(color=colors[1], width=2)
        ),
        row=1, col=1
    )

    # Plot population distribution if requested
    if show_population:
        fig.add_trace(
            go.Histogram(
                x=pop_data,
                name='Population',
                marker_color=colors[0],
                opacity=0.7,
                histnorm='probability density'
            ),
            row=1, col=2
        )

    fig.update_layout(
        title=f"CLT Demonstration: {pop_name}, n={sample_size}",
        showlegend=True,
        bargap=0.1,
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

    # Statistics display
    st.subheader("Key Statistics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Population Mean (μ)", f"{true_mean:.4f}")
        st.metric("Mean of Sample Means", f"{np.mean(sample_means):.4f}")
    with col2:
        st.metric("Population SD (σ)", f"{true_std:.4f}")
        st.metric("SD of Sample Means", f"{np.std(sample_means):.4f}")
    with col3:
        st.metric("Theoretical SE (σ/√n)", f"{true_std / np.sqrt(sample_size):.4f}")
        st.metric("Empirical SE", f"{np.std(pop_data) / np.sqrt(sample_size):.4f}")

    # QQ plot for normality check
    st.subheader("Normality Assessment (Q-Q Plot)")
    qq = stats.probplot(sample_means, dist="norm")
    x_qq = np.array([qq[0][0][0], qq[0][0][-1]])

    fig_qq = go.Figure()
    fig_qq.add_trace(go.Scatter(
        x=qq[0][0],
        y=qq[0][1],
        mode='markers',
        name='Sample Quantiles',
        marker=dict(color=colors[0])
    ))
    fig_qq.add_trace(go.Scatter(
        x=x_qq,
        y=qq[1][1] + qq[1][0] * x_qq,
        mode='lines',
        name='Normal Reference',
        line=dict(color=colors[1])
    ))
    fig_qq.update_layout(
        title="Q-Q Plot of Sample Means",
        xaxis_title="Theoretical Quantiles",
        yaxis_title="Sample Quantiles"
    )
    st.plotly_chart(fig_qq, use_container_width=True)

    # Interpretation
    st.subheader("Interpretation")
    if sample_size >= 30:
        st.success("""
        The sampling distribution approximates a normal distribution, as predicted by the CLT.
        The empirical standard error closely matches the theoretical prediction (σ/√n).
        """)
    else:
        st.warning("""
        With small sample sizes, the normal approximation may not be perfect. 
        For highly non-normal populations, consider larger samples (n ≥ 30).
        """)


def descriptive_statistics(numeric_cols):
    st.header("Descriptive Statistics")

    st.markdown("""
    <div class="test-info">
    <h4>About Descriptive Statistics</h4>
    <p>Descriptive statistics summarize and describe the main features of a dataset. They provide simple summaries about the sample and the measures.</p>
    <p><strong>Common measures include:</strong></p>
    <ul>
        <li><strong>Central tendency:</strong> Mean, Median, Mode</li>
        <li><strong>Dispersion:</strong> Range, Variance, Standard Deviation</li>
        <li><strong>Shape:</strong> Skewness, Kurtosis</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="example-box">
    <h4>Example Use Cases:</h4>
    <ul>
        <li>Understanding the average income of a population</li>
        <li>Analyzing the spread of test scores in a class</li>
        <li>Examining the distribution of product prices</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    if not numeric_cols:
        st.warning("No numeric columns found for descriptive statistics.")
        return

    selected_cols = st.multiselect("Select columns for analysis", numeric_cols, default=numeric_cols)

    if selected_cols:
        desc_stats = st.session_state.df[selected_cols].describe().T
        desc_stats['skewness'] = st.session_state.df[selected_cols].skew()
        desc_stats['kurtosis'] = st.session_state.df[selected_cols].kurtosis()
        desc_stats['missing'] = st.session_state.df[selected_cols].isna().sum()

        st.subheader("Summary Statistics")
        st.dataframe(desc_stats)

        st.subheader("Visualizations")

        col1, col2 = st.columns(2)
        with col1:
            plot_type = st.selectbox("Select plot type", ["Histogram", "Boxplot", "Violin Plot", "Density Plot"])

        with col2:
            selected_col = st.selectbox("Select column to visualize", selected_cols)

        colors = get_colors(st.session_state.color_palette, 1)

        if plot_type == "Histogram":
            fig = px.histogram(st.session_state.df, x=selected_col, marginal="box",
                               title=f"Distribution of {selected_col}",
                               color_discrete_sequence=colors)
            st.plotly_chart(fig, use_container_width=True)

        elif plot_type == "Boxplot":
            fig = px.box(st.session_state.df, y=selected_col, title=f"Boxplot of {selected_col}",
                         color_discrete_sequence=colors)
            st.plotly_chart(fig, use_container_width=True)

        elif plot_type == "Violin Plot":
            fig = px.violin(st.session_state.df, y=selected_col, title=f"Violin Plot of {selected_col}",
                            color_discrete_sequence=colors)
            st.plotly_chart(fig, use_container_width=True)

        elif plot_type == "Density Plot":
            fig = px.density_contour(st.session_state.df, x=selected_col, title=f"Density Plot of {selected_col}",
                                     color_discrete_sequence=colors)
            st.plotly_chart(fig, use_container_width=True)


def normality_tests(numeric_cols):
    st.header("Normality Tests")

    st.markdown("""
    <div class="test-info">
    <h4>About Normality Tests</h4>
    <p>Normality tests assess whether a dataset is well-modeled by a normal distribution. Many statistical tests assume normality.</p>
    <p><strong>Common tests include:</strong></p>
    <ul>
        <li><strong>Shapiro-Wilk:</strong> Good for small to medium samples (n < 50)</li>
        <li><strong>Kolmogorov-Smirnov:</strong> Compares sample with reference distribution</li>
        <li><strong>Anderson-Darling:</strong> More sensitive to tails of distribution</li>
    </ul>
    <p><strong>Interpretation:</strong> If p-value < 0.05, we reject the null hypothesis of normality.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="example-box">
    <h4>Example Use Cases:</h4>
    <ul>
        <li>Checking assumptions before parametric tests (t-tests, ANOVA)</li>
        <li>Validating normality for quality control processes</li>
        <li>Assessing distribution of residuals in regression analysis</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    if not numeric_cols:
        st.warning("No numeric columns found for normality tests.")
        return

    selected_col = st.selectbox("Select column to test for normality", numeric_cols)

    st.subheader("Graphical Assessment")

    colors = get_colors(st.session_state.color_palette, 2)

    # Create separate figures instead of subplots
    col1, col2 = st.columns(2)

    with col1:
        # Histogram with box plot
        hist_fig = px.histogram(st.session_state.df, x=selected_col, nbins=30,
                                marginal="box", title=f"Histogram of {selected_col}",
                                color_discrete_sequence=[colors[0]])
        st.plotly_chart(hist_fig, use_container_width=True)

    with col2:
        # Q-Q plot
        qq = stats.probplot(st.session_state.df[selected_col].dropna(), dist="norm")
        x = np.array([qq[0][0][0], qq[0][0][-1]])

        qq_fig = go.Figure()
        qq_fig.add_trace(go.Scatter(
            x=qq[0][0],
            y=qq[0][1],
            mode='markers',
            name='Data',
            marker=dict(color=colors[0])
        ))
        qq_fig.add_trace(go.Scatter(
            x=x,
            y=qq[1][1] + qq[1][0] * x,
            mode='lines',
            name='Normal',
            line=dict(color=colors[1])
        ))
        qq_fig.update_layout(
            title=f"Q-Q Plot of {selected_col}",
            xaxis_title="Theoretical Quantiles",
            yaxis_title="Sample Quantiles"
        )
        st.plotly_chart(qq_fig, use_container_width=True)

    st.subheader("Statistical Tests")

    # Remove missing values
    data = st.session_state.df[selected_col].dropna()

    # Shapiro-Wilk test
    shapiro_stat, shapiro_p = stats.shapiro(data)
    st.write(f"**Shapiro-Wilk Test**: W = {shapiro_stat:.4f}, p = {shapiro_p:.4f}")

    # Kolmogorov-Smirnov test
    ks_stat, ks_p = stats.kstest(data, 'norm', args=(data.mean(), data.std()))
    st.write(f"**Kolmogorov-Smirnov Test**: D = {ks_stat:.4f}, p = {ks_p:.4f}")

    # Anderson-Darling test
    anderson_result = stats.anderson(data, dist='norm')
    st.write(f"**Anderson-Darling Test**: A² = {anderson_result.statistic:.4f}")
    st.write("Critical values:")
    for i in range(len(anderson_result.critical_values)):
        sl, cv = anderson_result.significance_level[i], anderson_result.critical_values[i]
        st.write(f"{sl}%: {cv:.4f} - {'Normal' if anderson_result.statistic < cv else 'Not normal'}")


def t_tests(numeric_cols):
    st.header("T-tests")

    st.markdown("""
    <div class="test-info">
    <h4>About T-tests</h4>
    <p>T-tests compare means between groups to determine if they are statistically different.</p>
    <p><strong>Types of t-tests:</strong></p>
    <ul>
        <li><strong>One-sample:</strong> Compare sample mean to a known value</li>
        <li><strong>Independent two-sample:</strong> Compare means between two independent groups</li>
        <li><strong>Paired:</strong> Compare means from the same group at different times</li>
    </ul>
    <p><strong>Assumptions:</strong> Normality, equal variance (for independent t-test), interval/ratio data</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="example-box">
    <h4>Example Use Cases:</h4>
    <ul>
        <li>Comparing test scores between two classrooms (independent)</li>
        <li>Testing if average height differs from national average (one-sample)</li>
        <li>Measuring effect of training program with pre/post tests (paired)</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    if not numeric_cols:
        st.warning("No numeric columns found for t-tests.")
        return

    test_type = st.radio(
        "Select t-test type",
        ["One-sample t-test", "Independent two-sample t-test", "Paired t-test"]
    )

    colors = get_colors(st.session_state.color_palette, 3)

    if test_type == "One-sample t-test":
        selected_col = st.selectbox("Select column for one-sample test", numeric_cols)
        test_value = st.number_input("Enter test value (population mean)", value=0.0)

        data = st.session_state.df[selected_col].dropna()
        t_stat, p_value = stats.ttest_1samp(data, test_value)

        st.subheader("Results")
        st.write(f"**Test statistic (t)**: {t_stat:.4f}")
        st.write(f"**p-value**: {p_value:.4f}")
        st.write(f"**Degrees of freedom**: {len(data) - 1}")
        st.write(f"**Sample mean**: {data.mean():.4f}")
        st.write(f"**Sample size**: {len(data)}")

        # Plot
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=data, name='Sample Distribution', marker_color=colors[0]))
        fig.add_vline(x=test_value, line_dash="dash", line_color=colors[1], annotation_text="Hypothesized Mean")
        fig.add_vline(x=data.mean(), line_dash="dash", line_color=colors[2], annotation_text="Sample Mean")
        fig.update_layout(title=f"Distribution of {selected_col} with Hypothesized Mean")
        st.plotly_chart(fig, use_container_width=True)

    elif test_type == "Independent two-sample t-test":
        col1, col2 = st.columns(2)
        with col1:
            selected_col = st.selectbox("Select numeric column", numeric_cols)
        with col2:
            group_col = st.selectbox("Select grouping column",
                                     [c for c in st.session_state.df.columns if c not in numeric_cols])

        if st.session_state.df[group_col].nunique() != 2:
            st.error("Grouping column must have exactly 2 categories for independent t-test.")
            return

        groups = st.session_state.df[group_col].unique()
        group1, group2 = groups[0], groups[1]

        data1 = st.session_state.df[st.session_state.df[group_col] == group1][selected_col].dropna()
        data2 = st.session_state.df[st.session_state.df[group_col] == group2][selected_col].dropna()

        # Check for equal variances
        levene_stat, levene_p = stats.levene(data1, data2)
        equal_var = True if levene_p > 0.05 else False

        t_stat, p_value = stats.ttest_ind(data1, data2, equal_var=equal_var)

        st.subheader("Results")
        st.write(f"**Test statistic (t)**: {t_stat:.4f}")
        st.write(f"**p-value**: {p_value:.4f}")
        st.write(f"**Degrees of freedom**: {len(data1) + len(data2) - 2}")
        st.write(f"**Group '{group1}' mean**: {data1.mean():.4f} (n={len(data1)})")
        st.write(f"**Group '{group2}' mean**: {data2.mean():.4f} (n={len(data2)})")
        st.write(f"**Equal variances assumed**: {equal_var} (Levene's test p={levene_p:.4f})")

        # Plot
        fig = go.Figure()
        fig.add_trace(go.Box(y=data1, name=group1, boxpoints='all', jitter=0.3, pointpos=-1.8,
                             marker_color=colors[0]))
        fig.add_trace(go.Box(y=data2, name=group2, boxpoints='all', jitter=0.3, pointpos=1.8,
                             marker_color=colors[1]))
        fig.update_layout(title=f"Comparison of {selected_col} between {group1} and {group2}")
        st.plotly_chart(fig, use_container_width=True)

    elif test_type == "Paired t-test":
        col1, col2 = st.columns(2)
        with col1:
            selected_col1 = st.selectbox("Select first measurement", numeric_cols)
        with col2:
            selected_col2 = st.selectbox("Select second measurement", numeric_cols)

        paired_data = st.session_state.df[[selected_col1, selected_col2]].dropna()
        t_stat, p_value = stats.ttest_rel(paired_data[selected_col1], paired_data[selected_col2])

        st.subheader("Results")
        st.write(f"**Test statistic (t)**: {t_stat:.4f}")
        st.write(f"**p-value**: {p_value:.4f}")
        st.write(f"**Degrees of freedom**: {len(paired_data) - 1}")
        st.write(f"**Mean difference**: {(paired_data[selected_col1] - paired_data[selected_col2]).mean():.4f}")
        st.write(f"**Sample size**: {len(paired_data)}")

        # Plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=paired_data[selected_col1], y=paired_data[selected_col2],
                                 mode='markers', name='Data points', marker_color=colors[0]))
        fig.add_trace(go.Scatter(x=[min(paired_data.min()), max(paired_data.max())],
                                 y=[min(paired_data.min()), max(paired_data.max())],
                                 mode='lines', name='Identity line', line_color=colors[1]))
        fig.update_layout(title=f"Paired Comparison: {selected_col1} vs {selected_col2}",
                          xaxis_title=selected_col1, yaxis_title=selected_col2)
        st.plotly_chart(fig, use_container_width=True)

        # Plot differences
        fig2 = go.Figure()
        fig2.add_trace(go.Histogram(x=paired_data[selected_col1] - paired_data[selected_col2],
                                    name='Differences', marker_color=colors[2]))
        fig2.add_vline(x=0, line_dash="dash", line_color=colors[1], annotation_text="No difference")
        fig2.update_layout(title=f"Distribution of Differences ({selected_col1} - {selected_col2})")
        st.plotly_chart(fig2, use_container_width=True)


def anova_tests(numeric_cols, categorical_cols):
    st.header("ANOVA Tests")

    st.markdown("""
    <div class="test-info">
    <h4>About ANOVA</h4>
    <p>Analysis of Variance (ANOVA) tests differences among group means in a sample.</p>
    <p><strong>Types of ANOVA:</strong></p>
    <ul>
        <li><strong>One-way:</strong> Single factor with multiple levels</li>
        <li><strong>Two-way:</strong> Two factors with interaction effects</li>
        <li><strong>Repeated Measures:</strong> Same subjects across conditions/time</li>
    </ul>
    <p><strong>Assumptions:</strong> Normality, homogeneity of variance, independence (for between-subjects)</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="example-box">
    <h4>Example Use Cases:</h4>
    <ul>
        <li>Comparing effectiveness of 3+ teaching methods (one-way)</li>
        <li>Testing drug effects across genders (two-way)</li>
        <li>Measuring learning improvement across 4 tests (repeated measures)</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    if not numeric_cols or not categorical_cols:
        st.warning("Need both numeric and categorical columns for ANOVA.")
        return

    test_type = st.radio(
        "Select ANOVA type",
        ["One-way ANOVA", "Two-way ANOVA", "Repeated Measures ANOVA"]
    )

    colors = get_colors(st.session_state.color_palette, 10)

    if test_type == "One-way ANOVA":
        col1, col2 = st.columns(2)
        with col1:
            selected_col = st.selectbox("Select numeric column", numeric_cols)
        with col2:
            group_col = st.selectbox("Select grouping column", categorical_cols)

        if st.session_state.df[group_col].nunique() < 2:
            st.error("Grouping column must have at least 2 categories for ANOVA.")
            return

        # Prepare data for ANOVA
        groups = st.session_state.df[group_col].unique()
        group_data = [st.session_state.df[st.session_state.df[group_col] == g][selected_col].dropna() for g in groups]

        # Perform ANOVA
        f_stat, p_value = stats.f_oneway(*group_data)

        st.subheader("Results")
        st.write(f"**F-statistic**: {f_stat:.4f}")
        st.write(f"**p-value**: {p_value:.4f}")

        # Plot group distributions
        fig = px.box(st.session_state.df, x=group_col, y=selected_col, color=group_col,
                     title=f"Distribution of {selected_col} by {group_col}",
                     color_discrete_sequence=colors)
        st.plotly_chart(fig, use_container_width=True)

        # Post-hoc tests if significant
        if p_value < 0.05:
            st.subheader("Post-hoc Tests (Tukey HSD)")
            tukey_result = pairwise_tukeyhsd(
                endog=st.session_state.df[selected_col].dropna(),
                groups=st.session_state.df[group_col].dropna(),
                alpha=0.05
            )

            # Create a table for display
            tukey_df = pd.DataFrame(data=tukey_result._results_table.data[1:],
                                    columns=tukey_result._results_table.data[0])
            st.dataframe(tukey_df)

            # Plot the results
            fig2 = go.Figure()
            for i, (group1, group2) in enumerate(zip(tukey_result.groupsunique[tukey_result._results_table.data[1:][0]],
                                                     tukey_result.groupsunique[
                                                         tukey_result._results_table.data[1:][1]])):
                fig2.add_trace(go.Scatter(
                    x=[tukey_result._results_table.data[1:][i][2], tukey_result._results_table.data[1:][i][3]],
                    y=[f"{group1}-{group2}", f"{group1}-{group2}"],
                    mode="lines",
                    line=dict(color=colors[i % len(colors)], width=2),
                    showlegend=False
                ))
                fig2.add_trace(go.Scatter(
                    x=[tukey_result._results_table.data[1:][i][1]],
                    y=[f"{group1}-{group2}"],
                    mode="markers",
                    marker=dict(color=colors[i % len(colors)], size=10),
                    showlegend=False
                ))

            fig2.update_layout(
                title="Tukey HSD Results",
                xaxis_title="Mean Difference",
                yaxis_title="Group Comparison",
                showlegend=False
            )
            st.plotly_chart(fig2, use_container_width=True)

    elif test_type == "Two-way ANOVA":
        st.subheader("Two-way ANOVA")
        col1, col2, col3 = st.columns(3)
        with col1:
            selected_col = st.selectbox("Select dependent variable", numeric_cols)
        with col2:
            factor1 = st.selectbox("Select first factor", categorical_cols)
        with col3:
            factor2 = st.selectbox("Select second factor", categorical_cols)

        try:
            # Create a clean formula without special characters
            # First ensure column names are valid Python identifiers
            df_clean = st.session_state.df.dropna().copy()
            df_clean.columns = [''.join(c if c.isalnum() else '_' for c in str(col)) for col in df_clean.columns]

            # Map back to clean column names
            selected_col_clean = ''.join(c if c.isalnum() else '_' for c in str(selected_col))
            factor1_clean = ''.join(c if c.isalnum() else '_' for c in str(factor1))
            factor2_clean = ''.join(c if c.isalnum() else '_' for c in str(factor2))

            # Prepare formula
            formula = f"{selected_col_clean} ~ {factor1_clean} + {factor2_clean} + {factor1_clean}:{factor2_clean}"

            # Fit model
            model = ols(formula, data=df_clean).fit()
            anova_table = anova_lm(model, typ=2)

            st.subheader("ANOVA Table")
            st.dataframe(anova_table)

            # Interaction plot if significant interaction
            interaction_term = f"{factor1_clean}:{factor2_clean}"
            if interaction_term in anova_table.index and anova_table.loc[interaction_term, 'PR(>F)'] < 0.05:
                st.subheader("Interaction Plot")
                fig = px.line(st.session_state.df.dropna(), x=factor1, y=selected_col, color=factor2,
                              title=f"Interaction Plot: {selected_col} by {factor1} and {factor2}",
                              color_discrete_sequence=colors)
                st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Error in Two-way ANOVA: {str(e)}")
            st.error("Please check that your column names don't contain special characters or spaces.")
            st.error("Try renaming columns to use only letters, numbers, and underscores.")

    elif test_type == "Repeated Measures ANOVA":
        st.subheader("Repeated Measures ANOVA")
        st.info("This requires data in long format with a subject ID column.")

        subject_col = st.selectbox("Select subject ID column", st.session_state.df.columns)
        within_factor = st.selectbox("Select within-subjects factor", categorical_cols)
        dv = st.selectbox("Select dependent variable", numeric_cols)

        if st.session_state.df[within_factor].nunique() < 2:
            st.error("Within-subjects factor must have at least 2 levels.")
            return

        try:
            # Use pingouin for RM ANOVA
            aov = pg.rm_anova(data=st.session_state.df, dv=dv, within=within_factor,
                              subject=subject_col, detailed=True)

            st.subheader("Results")
            st.dataframe(aov)

            # Plot the data
            fig = px.box(st.session_state.df, x=within_factor, y=dv, color=within_factor,
                         title=f"Distribution of {dv} by {within_factor}",
                         color_discrete_sequence=colors)
            st.plotly_chart(fig, use_container_width=True)

            # Post-hoc pairwise comparisons
            posthoc = pg.pairwise_ttests(data=st.session_state.df, dv=dv, within=[within_factor],
                                         subject=subject_col, padjust='bonf')
            st.subheader("Post-hoc Pairwise Comparisons")
            st.dataframe(posthoc)

        except Exception as e:
            st.error(f"Error in Repeated Measures ANOVA: {str(e)}")


def correlation_analysis(numeric_cols):
    st.header("Correlation Analysis")

    st.markdown("""
    <div class="test-info">
    <h4>About Correlation Analysis</h4>
    <p>Correlation measures the statistical relationship between two variables.</p>
    <p><strong>Types of correlation:</strong></p>
    <ul>
        <li><strong>Pearson:</strong> Linear relationship between continuous variables</li>
        <li><strong>Spearman:</strong> Monotonic relationship (ordinal/continuous)</li>
        <li><strong>Kendall's Tau:</strong> Similar to Spearman, better for small samples</li>
    </ul>
    <p><strong>Interpretation:</strong> Values range from -1 (perfect negative) to +1 (perfect positive)</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="example-box">
    <h4>Example Use Cases:</h4>
    <ul>
        <li>Examining relationship between study time and test scores</li>
        <li>Analyzing association between temperature and sales</li>
        <li>Investigating link between age and blood pressure</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    if len(numeric_cols) < 2:
        st.warning("Need at least 2 numeric columns for correlation analysis.")
        return

    selected_cols = st.multiselect("Select columns for correlation", numeric_cols, default=numeric_cols[:2])

    if len(selected_cols) < 2:
        st.warning("Please select at least 2 columns.")
        return

    st.subheader("Correlation Matrix")
    corr_matrix = st.session_state.df[selected_cols].corr()
    st.dataframe(corr_matrix)

    # Plot correlation heatmap
    colors = get_colors(st.session_state.color_palette, 2)  # Get two colors for the heatmap

    # Create a custom colorscale from our palette
    colorscale = [
        [0.0, colors[0]],
        [1.0, colors[1]]
    ]

    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale=colorscale,
        zmin=-1,
        zmax=1,
        text=corr_matrix.round(2).values,
        texttemplate="%{text}",
        hoverinfo="x+y+z"
    ))

    fig.update_layout(
        title="Correlation Heatmap",
        xaxis_title="Variables",
        yaxis_title="Variables",
        width=800,
        height=600
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Pairwise Scatter Plots")
    fig = px.scatter_matrix(st.session_state.df[selected_cols],
                            color_discrete_sequence=get_colors(st.session_state.color_palette, len(selected_cols)))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Statistical Tests")
    col1, col2 = st.columns(2)
    with col1:
        var1 = st.selectbox("Select first variable", selected_cols)
    with col2:
        var2 = st.selectbox("Select second variable", [c for c in selected_cols if c != var1])

    # Pearson correlation
    pearson_r, pearson_p = stats.pearsonr(st.session_state.df[var1].dropna(), st.session_state.df[var2].dropna())
    st.write(f"**Pearson correlation**: r = {pearson_r:.4f}, p = {pearson_p:.4f}")

    # Spearman correlation
    spearman_r, spearman_p = stats.spearmanr(st.session_state.df[var1].dropna(), st.session_state.df[var2].dropna())
    st.write(f"**Spearman correlation**: ρ = {spearman_r:.4f}, p = {spearman_p:.4f}")

    # Kendall's tau
    kendall_tau, kendall_p = stats.kendalltau(st.session_state.df[var1].dropna(), st.session_state.df[var2].dropna())
    st.write(f"**Kendall's tau**: τ = {kendall_tau:.4f}, p = {kendall_p:.4f}")

    # Scatter plot with regression line
    colors = get_colors(st.session_state.color_palette, 1)
    fig = px.scatter(st.session_state.df, x=var1, y=var2, trendline="ols",
                     title=f"Scatter Plot of {var1} vs {var2} with Regression Line",
                     color_discrete_sequence=colors)
    st.plotly_chart(fig, use_container_width=True)


def chi_square_tests(categorical_cols):
    st.header("Chi-square Tests")

    st.markdown("""
    <div class="test-info">
    <h4>About Chi-square Tests</h4>
    <p>Chi-square tests examine relationships between categorical variables.</p>
    <p><strong>Types of tests:</strong></p>
    <ul>
        <li><strong>Goodness-of-fit:</strong> Compare observed to expected distribution</li>
        <li><strong>Test of independence:</strong> Check association between two variables</li>
    </ul>
    <p><strong>Assumptions:</strong> Expected frequencies ≥5 in 80% of cells, independent observations</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="example-box">
    <h4>Example Use Cases:</h4>
    <ul>
        <li>Testing if gender distribution matches population (goodness-of-fit)</li>
        <li>Examining relationship between education level and voting preference (independence)</li>
        <li>Checking if product preference varies by region</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    if len(categorical_cols) < 2:
        st.warning("Need at least 2 categorical columns for chi-square tests.")
        return

    test_type = st.radio(
        "Select chi-square test type",
        ["Goodness-of-fit test", "Test of independence"]
    )

    colors = get_colors(st.session_state.color_palette, 2)

    if test_type == "Goodness-of-fit test":
        selected_col = st.selectbox("Select categorical column", categorical_cols)

        # Get observed frequencies
        observed = st.session_state.df[selected_col].value_counts().sort_index()
        st.subheader("Observed Frequencies")
        st.dataframe(observed)

        # Expected frequencies options
        expected_option = st.radio(
            "Expected frequencies",
            ["Equal proportions", "Custom proportions"]
        )

        if expected_option == "Equal proportions":
            expected = np.ones(len(observed)) * (len(st.session_state.df) / len(observed))
        else:
            expected = []
            for cat in observed.index:
                val = st.number_input(f"Expected proportion for '{cat}'", min_value=0.0, max_value=1.0,
                                      value=1.0 / len(observed))
                expected.append(val * len(st.session_state.df))
            expected = np.array(expected)

        # Perform chi-square test
        chi2_stat, p_value = stats.chisquare(f_obs=observed, f_exp=expected)

        st.subheader("Results")
        st.write(f"**Chi-square statistic**: {chi2_stat:.4f}")
        st.write(f"**p-value**: {p_value:.4f}")
        st.write(f"**Degrees of freedom**: {len(observed) - 1}")

        # Plot observed vs expected
        fig = go.Figure()
        fig.add_trace(go.Bar(x=observed.index, y=observed.values, name='Observed', marker_color=colors[0]))
        fig.add_trace(go.Bar(x=observed.index, y=expected, name='Expected', marker_color=colors[1]))
        fig.update_layout(barmode='group', title="Observed vs Expected Frequencies")
        st.plotly_chart(fig, use_container_width=True)

    elif test_type == "Test of independence":
        col1, col2 = st.columns(2)
        with col1:
            var1 = st.selectbox("Select first variable", categorical_cols)
        with col2:
            var2 = st.selectbox("Select second variable", [c for c in categorical_cols if c != var1])

        # Create contingency table
        contingency_table = pd.crosstab(st.session_state.df[var1], st.session_state.df[var2])
        st.subheader("Contingency Table")
        st.dataframe(contingency_table)

        # Plot stacked bar chart
        fig = px.bar(contingency_table, barmode='stack',
                     title=f"Distribution of {var1} by {var2}",
                     color_discrete_sequence=colors)
        st.plotly_chart(fig, use_container_width=True)

        # Perform chi-square test
        chi2_stat, p_value, dof, expected = stats.chi2_contingency(contingency_table)

        st.subheader("Results")
        st.write(f"**Chi-square statistic**: {chi2_stat:.4f}")
        st.write(f"**p-value**: {p_value:.4f}")
        st.write(f"**Degrees of freedom**: {dof}")

        # Show expected frequencies if significant
        if p_value < 0.05:
            st.subheader("Expected Frequencies (if independent)")
            st.dataframe(pd.DataFrame(expected, index=contingency_table.index, columns=contingency_table.columns))


def nonparametric_tests(numeric_cols, categorical_cols):
    st.header("Non-parametric Tests")

    st.markdown("""
    <div class="test-info">
    <h4>About Non-parametric Tests</h4>
    <p>Non-parametric tests don't assume normal distribution and are used when parametric assumptions are violated.</p>
    <p><strong>Common tests:</strong></p>
    <ul>
        <li><strong>Mann-Whitney U:</strong> Non-parametric alternative to independent t-test</li>
        <li><strong>Wilcoxon signed-rank:</strong> Alternative to paired t-test</li>
        <li><strong>Kruskal-Wallis:</strong> Alternative to one-way ANOVA</li>
        <li><strong>Friedman test:</strong> Alternative to repeated measures ANOVA</li>
    </ul>
    <p><strong>When to use:</strong> Small samples, ordinal data, non-normal distributions</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="example-box">
    <h4>Example Use Cases:</h4>
    <ul>
        <li>Comparing customer satisfaction ratings (ordinal data)</li>
        <li>Analyzing reaction times (typically skewed)</li>
        <li>Testing small samples where normality can't be verified</li>
        <li>Working with ranked data or outliers</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    if not numeric_cols:
        st.warning("No numeric columns found for non-parametric tests.")
        return

    test_type = st.radio(
        "Select non-parametric test",
        [
            "Mann-Whitney U (Wilcoxon rank-sum)",
            "Wilcoxon signed-rank",
            "Kruskal-Wallis",
            "Friedman test"
        ]
    )

    colors = get_colors(st.session_state.color_palette, 10)

    if test_type == "Mann-Whitney U (Wilcoxon rank-sum)":
        col1, col2 = st.columns(2)
        with col1:
            selected_col = st.selectbox("Select numeric column", numeric_cols)
        with col2:
            group_col = st.selectbox("Select grouping column",
                                     [c for c in st.session_state.df.columns if c not in numeric_cols])

        if st.session_state.df[group_col].nunique() != 2:
            st.error("Grouping column must have exactly 2 categories for Mann-Whitney U test.")
            return

        groups = st.session_state.df[group_col].unique()
        group1, group2 = groups[0], groups[1]

        data1 = st.session_state.df[st.session_state.df[group_col] == group1][selected_col].dropna()
        data2 = st.session_state.df[st.session_state.df[group_col] == group2][selected_col].dropna()

        u_stat, p_value = stats.mannwhitneyu(data1, data2)

        st.subheader("Results")
        st.write(f"**U statistic**: {u_stat:.4f}")
        st.write(f"**p-value**: {p_value:.4f}")
        st.write(f"**Group '{group1}' median**: {data1.median():.4f} (n={len(data1)})")
        st.write(f"**Group '{group2}' median**: {data2.median():.4f} (n={len(data2)})")

        # Plot
        fig = go.Figure()
        fig.add_trace(go.Box(y=data1, name=group1, boxpoints='all', jitter=0.3, pointpos=-1.8,
                             marker_color=colors[0]))
        fig.add_trace(go.Box(y=data2, name=group2, boxpoints='all', jitter=0.3, pointpos=1.8,
                             marker_color=colors[1]))
        fig.update_layout(title=f"Comparison of {selected_col} between {group1} and {group2}")
        st.plotly_chart(fig, use_container_width=True)

    elif test_type == "Wilcoxon signed-rank":
        col1, col2 = st.columns(2)
        with col1:
            selected_col1 = st.selectbox("Select first measurement", numeric_cols)
        with col2:
            selected_col2 = st.selectbox("Select second measurement", numeric_cols)

        paired_data = st.session_state.df[[selected_col1, selected_col2]].dropna()
        stat, p_value = stats.wilcoxon(paired_data[selected_col1], paired_data[selected_col2])

        st.subheader("Results")
        st.write(f"**Test statistic**: {stat:.4f}")
        st.write(f"**p-value**: {p_value:.4f}")
        st.write(f"**Median difference**: {(paired_data[selected_col1] - paired_data[selected_col2]).median():.4f}")
        st.write(f"**Sample size**: {len(paired_data)}")

        # Plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=paired_data[selected_col1], y=paired_data[selected_col2],
                                 mode='markers', name='Data points', marker_color=colors[0]))
        fig.add_trace(go.Scatter(x=[min(paired_data.min()), max(paired_data.max())],
                                 y=[min(paired_data.min()), max(paired_data.max())],
                                 mode='lines', name='Identity line', line_color=colors[1]))
        fig.update_layout(title=f"Paired Comparison: {selected_col1} vs {selected_col2}",
                          xaxis_title=selected_col1, yaxis_title=selected_col2)
        st.plotly_chart(fig, use_container_width=True)

        # Plot differences
        fig2 = go.Figure()
        fig2.add_trace(go.Histogram(x=paired_data[selected_col1] - paired_data[selected_col2],
                                    name='Differences', marker_color=colors[2]))
        fig2.add_vline(x=0, line_dash="dash", line_color=colors[1], annotation_text="No difference")
        fig2.update_layout(title=f"Distribution of Differences ({selected_col1} - {selected_col2})")
        st.plotly_chart(fig2, use_container_width=True)

    elif test_type == "Kruskal-Wallis":
        col1, col2 = st.columns(2)
        with col1:
            selected_col = st.selectbox("Select numeric column", numeric_cols)
        with col2:
            group_col = st.selectbox("Select grouping column",
                                     [c for c in st.session_state.df.columns if c not in numeric_cols])

        if st.session_state.df[group_col].nunique() < 2:
            st.error("Grouping column must have at least 2 categories for Kruskal-Wallis test.")
            return

        groups = st.session_state.df[group_col].unique()
        group_data = [st.session_state.df[st.session_state.df[group_col] == g][selected_col].dropna() for g in groups]

        h_stat, p_value = stats.kruskal(*group_data)

        st.subheader("Results")
        st.write(f"**H statistic**: {h_stat:.4f}")
        st.write(f"**p-value**: {p_value:.4f}")

        # Plot
        fig = px.box(st.session_state.df, x=group_col, y=selected_col, color=group_col,
                     title=f"Distribution of {selected_col} by {group_col}",
                     color_discrete_sequence=colors)
        st.plotly_chart(fig, use_container_width=True)

        # Post-hoc Dunn's test if significant
        if p_value < 0.05:
            st.subheader("Post-hoc Dunn's Test")
            try:
                dunn_result = pg.pairwise_ttests(
                    data=st.session_state.df.dropna(),
                    dv=selected_col,
                    between=group_col,
                    padjust='bonf'
                )
                st.dataframe(dunn_result)
            except Exception as e:
                st.error(f"Could not perform Dunn's test: {e}")

    elif test_type == "Friedman test":
        st.subheader("Friedman Test (Repeated Measures)")
        st.info("This requires data in wide format with each column representing a condition.")

        selected_cols = st.multiselect("Select measurements for each condition", numeric_cols)

        if len(selected_cols) < 2:
            st.warning("Please select at least 2 measurements.")
            return

        friedman_data = st.session_state.df[selected_cols].dropna()
        f_stat, p_value = stats.friedmanchisquare(*[friedman_data[col] for col in selected_cols])

        st.subheader("Results")
        st.write(f"**Friedman chi-square statistic**: {f_stat:.4f}")
        st.write(f"**p-value**: {p_value:.4f}")

        # Plot the data
        melted_data = pd.melt(friedman_data, var_name='Condition', value_name='Value')
        fig = px.box(melted_data, x='Condition', y='Value', color='Condition',
                     title="Distribution Across Conditions",
                     color_discrete_sequence=colors)
        st.plotly_chart(fig, use_container_width=True)

        # Post-hoc Nemenyi test if significant
        if p_value < 0.05 and len(selected_cols) > 2:
            st.subheader("Post-hoc Nemenyi Test")
            try:
                nemenyi_result = pg.pairwise_ttests(
                    data=st.session_state.df.melt(value_vars=selected_cols).dropna(),
                    dv='value',
                    within='variable',
                    parametric=False,
                    padjust='fdr_bh'
                )
                st.dataframe(nemenyi_result)
            except Exception as e:
                st.error(f"Could not perform Nemenyi test: {e}")


def regression_analysis(numeric_cols):
    st.header("Regression Analysis")

    st.markdown("""
    <div class="test-info">
    <h4>About Regression Analysis</h4>
    <p>Regression models the relationship between a dependent variable and one or more independent variables.</p>
    <p><strong>Types of regression:</strong></p>
    <ul>
        <li><strong>Simple Linear:</strong> One predictor variable</li>
        <li><strong>Multiple Linear:</strong> Multiple predictors</li>
        <li><strong>Logistic:</strong> For binary outcomes</li>
    </ul>
    <p><strong>Key outputs:</strong> Coefficients, R-squared, p-values, confidence intervals</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="example-box">
    <h4>Example Use Cases:</h4>
    <ul>
        <li>Predicting house prices based on features (multiple linear)</li>
        <li>Modeling the effect of study time on test scores (simple linear)</li>
        <li>Predicting likelihood of disease based on risk factors (logistic)</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    if len(numeric_cols) < 2:
        st.warning("Need at least 2 numeric columns for regression analysis.")
        return

    regression_type = st.radio(
        "Select regression type",
        ["Simple Linear Regression", "Multiple Linear Regression", "Logistic Regression"]
    )

    colors = get_colors(st.session_state.color_palette, 5)

    if regression_type == "Simple Linear Regression":
        col1, col2 = st.columns(2)
        with col1:
            x_var = st.selectbox("Select independent variable (X)", numeric_cols)
        with col2:
            y_var = st.selectbox("Select dependent variable (Y)", [c for c in numeric_cols if c != x_var])

        # Remove missing values
        data = st.session_state.df[[x_var, y_var]].dropna()
        x = data[x_var]
        y = data[y_var]

        # Fit regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

        st.subheader("Regression Results")
        st.write(f"**Regression equation**: Y = {intercept:.4f} + {slope:.4f} * X")
        st.write(f"**R-squared**: {r_value ** 2:.4f}")
        st.write(f"**p-value for slope**: {p_value:.4f}")
        st.write(f"**Standard error of estimate**: {std_err:.4f}")

        # Plot regression
        fig = px.scatter(data, x=x_var, y=y_var, trendline="ols",
                         title=f"Regression Plot: {y_var} vs {x_var}",
                         color_discrete_sequence=[colors[0]])
        st.plotly_chart(fig, use_container_width=True)

        # Residual analysis
        st.subheader("Residual Analysis")
        residuals = y - (intercept + slope * x)

        # Create subplots for residual analysis
        fig = make_subplots(rows=1, cols=3,
                            subplot_titles=("Residuals Distribution", "Q-Q Plot", "Residuals vs Fitted"))

        # Residuals histogram
        fig.add_trace(go.Histogram(x=residuals, name='Residuals', marker_color=colors[0]), row=1, col=1)

        # Q-Q plot of residuals
        qq = stats.probplot(residuals, dist="norm")
        x_qq = np.array([qq[0][0][0], qq[0][0][-1]])
        fig.add_trace(go.Scatter(x=qq[0][0], y=qq[0][1], mode='markers', name='Data',
                                 marker=dict(color=colors[1])), row=1, col=2)
        fig.add_trace(go.Scatter(x=x_qq, y=qq[1][1] + qq[1][0] * x_qq, mode='lines', name='Normal',
                                 line=dict(color=colors[2])), row=1, col=2)

        # Residuals vs fitted
        fitted = intercept + slope * x
        fig.add_trace(go.Scatter(x=fitted, y=residuals, mode='markers', name='Residuals',
                                 marker=dict(color=colors[3])), row=1, col=3)
        fig.add_hline(y=0, line_dash="dash", line_color=colors[4], row=1, col=3)

        fig.update_layout(title_text="Residual Analysis", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    elif regression_type == "Multiple Linear Regression":
        y_var = st.selectbox("Select dependent variable (Y)", numeric_cols)
        x_vars = st.multiselect("Select independent variables (X)", [c for c in numeric_cols if c != y_var])

        if not x_vars:
            st.warning("Please select at least one independent variable.")
            return

        # Prepare formula
        formula = f"{y_var} ~ {' + '.join(x_vars)}"

        # Fit model
        model = ols(formula, data=st.session_state.df.dropna()).fit()

        st.subheader("Regression Results")
        st.text(model.summary())

        # VIF for multicollinearity
        st.subheader("Variance Inflation Factors (VIF)")
        X = st.session_state.df[x_vars].dropna()
        vif_data = pd.DataFrame()
        vif_data["Variable"] = X.columns
        vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
        st.dataframe(vif_data)

        # Plot partial regression plots
        st.subheader("Partial Regression Plots")
        for x_var in x_vars:
            fig = px.scatter(st.session_state.df, x=x_var, y=y_var, trendline="ols",
                             title=f"Partial Regression: {y_var} vs {x_var}",
                             color_discrete_sequence=colors)
            st.plotly_chart(fig, use_container_width=True)

    elif regression_type == "Logistic Regression":
        st.subheader("Logistic Regression")
        st.info("For logistic regression, the dependent variable must be binary (0/1).")

        # Check for binary variables
        binary_cols = []
        for col in numeric_cols:
            unique_vals = st.session_state.df[col].dropna().unique()
            if len(unique_vals) == 2 and set(unique_vals).issubset({0, 1}):
                binary_cols.append(col)

        if not binary_cols:
            st.warning("No binary (0/1) variables found for logistic regression.")
            return

        y_var = st.selectbox("Select binary dependent variable", binary_cols)
        x_vars = st.multiselect("Select independent variables", [c for c in numeric_cols if c != y_var])

        if not x_vars:
            st.warning("Please select at least one independent variable.")
            return

        # Fit logistic regression
        try:
            X = st.session_state.df[x_vars].dropna()
            X = sm.add_constant(X)  # Add intercept
            y = st.session_state.df[y_var].dropna()

            # Ensure lengths match after dropping NA
            common_index = X.index.intersection(y.index)
            X = X.loc[common_index]
            y = y.loc[common_index]

            model = sm.Logit(y, X).fit(disp=0)

            st.subheader("Logistic Regression Results")
            st.text(model.summary())

            # Odds ratios
            st.subheader("Odds Ratios")
            odds_ratios = pd.DataFrame({
                "Variable": X.columns,
                "Odds Ratio": np.exp(model.params),
                "95% CI Lower": np.exp(model.conf_int()[0]),
                "95% CI Upper": np.exp(model.conf_int()[1])
            })
            st.dataframe(odds_ratios)

            # ROC curve
            st.subheader("Model Performance")
            y_pred = model.predict(X)
            fpr, tpr, thresholds = roc_curve(y, y_pred)
            auc = roc_auc_score(y, y_pred)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'AUC = {auc:.2f}',
                                     line=dict(color=colors[0], width=3)))
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line_dash='dash', name='Random',
                                     line=dict(color=colors[1])))
            fig.update_layout(title="ROC Curve",
                              xaxis_title="False Positive Rate",
                              yaxis_title="True Positive Rate")
            st.plotly_chart(fig, use_container_width=True)

            # Plot predicted probabilities
            fig2 = px.histogram(x=y_pred, color=y.astype(str), nbins=50,
                                title="Predicted Probabilities Distribution",
                                labels={'x': 'Predicted Probability', 'color': 'Actual Class'},
                                color_discrete_sequence=[colors[2], colors[3]])
            st.plotly_chart(fig2, use_container_width=True)

        except Exception as e:
            st.error(f"Error fitting logistic regression model: {e}")


def power_analysis():
    st.header("Power Analysis")

    st.markdown("""
    <div class="test-info">
    <h4>About Power Analysis</h4>
    <p>Power analysis helps determine sample sizes needed to detect effects and avoid Type II errors.</p>
    <p><strong>Key concepts:</strong></p>
    <ul>
        <li><strong>Power (1-β):</strong> Probability of detecting an effect if it exists (typically 0.8)</li>
        <li><strong>Effect size:</strong> Magnitude of the effect (Cohen's d, f, r)</li>
        <li><strong>Alpha (α):</strong> Significance level (typically 0.05)</li>
    </ul>
    <p><strong>Use:</strong> Planning studies, determining required sample sizes</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="example-box">
    <h4>Example Use Cases:</h4>
    <ul>
        <li>Determining how many participants needed for a clinical trial</li>
        <li>Assessing if a study has sufficient power to detect expected effects</li>
        <li>Planning research to ensure adequate sample sizes</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    test_type = st.radio(
        "Select test type for power analysis",
        ["T-test", "ANOVA", "Correlation", "Proportion test"]
    )

    colors = get_colors(st.session_state.color_palette, 2)

    if test_type == "T-test":
        col1, col2 = st.columns(2)
        with col1:
            effect_size = st.number_input("Effect size (Cohen's d)", min_value=0.01, value=0.5, step=0.1)
            alpha = st.number_input("Alpha level", min_value=0.001, max_value=0.2, value=0.05, step=0.01)
        with col2:
            desired_power = st.number_input("Desired power", min_value=0.1, max_value=0.99, value=0.8, step=0.05)
            ratio = st.number_input("Sample size ratio (n2/n1)", min_value=0.1, value=1.0, step=0.1)

        # Calculate required sample size
        analysis = sm_power.TTestPower()
        n = analysis.solve_power(effect_size=effect_size, power=desired_power, alpha=alpha, ratio=ratio)

        st.subheader("Results")
        st.write(f"**Required sample size per group**: {np.ceil(n):.0f}")
        st.write(f"**Total required sample size**: {np.ceil(n) * (1 + ratio):.0f}")

        # Plot power curve
        sample_sizes = np.arange(5, 200, 5)
        powers = [analysis.solve_power(effect_size=effect_size, nobs1=n, alpha=alpha, ratio=ratio) for n in
                  sample_sizes]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sample_sizes, y=powers, mode='lines', name='Power Curve',
                                 line=dict(color=colors[0], width=3)))
        fig.add_hline(y=desired_power, line_dash="dash", line_color=colors[1],
                      annotation_text=f"Target Power ({desired_power})")
        fig.update_layout(title="Power Curve",
                          xaxis_title="Sample Size per Group",
                          yaxis_title="Power")
        st.plotly_chart(fig, use_container_width=True)

    elif test_type == "ANOVA":
        col1, col2 = st.columns(2)
        with col1:
            effect_size = st.number_input("Effect size (f)", min_value=0.01, value=0.25, step=0.05)
            alpha = st.number_input("Alpha level", min_value=0.001, max_value=0.2, value=0.05, step=0.01)
        with col2:
            desired_power = st.number_input("Desired power", min_value=0.1, max_value=0.99, value=0.8, step=0.05)
            k_groups = st.number_input("Number of groups", min_value=2, value=3, step=1)

        # Calculate required sample size
        analysis = sm_power.FTestAnovaPower()
        n = analysis.solve_power(effect_size=effect_size, power=desired_power, alpha=alpha, k_groups=k_groups)

        st.subheader("Results")
        st.write(f"**Required sample size per group**: {np.ceil(n):.0f}")
        st.write(f"**Total required sample size**: {np.ceil(n) * k_groups:.0f}")

        # Plot power curve
        sample_sizes = np.arange(5, 200, 5)
        powers = [analysis.solve_power(effect_size=effect_size, nobs=n, alpha=alpha, k_groups=k_groups) for n in
                  sample_sizes]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sample_sizes, y=powers, mode='lines', name='Power Curve',
                                 line=dict(color=colors[0], width=3)))
        fig.add_hline(y=desired_power, line_dash="dash", line_color=colors[1],
                      annotation_text=f"Target Power ({desired_power})")
        fig.update_layout(title="Power Curve",
                          xaxis_title="Sample Size per Group",
                          yaxis_title="Power")
        st.plotly_chart(fig, use_container_width=True)

    elif test_type == "Correlation":
        col1, col2 = st.columns(2)
        with col1:
            effect_size = st.number_input("Effect size (r)", min_value=0.01, max_value=0.99, value=0.3, step=0.05)
            alpha = st.number_input("Alpha level", min_value=0.001, max_value=0.2, value=0.05, step=0.01)
        with col2:
            desired_power = st.number_input("Desired power", min_value=0.1, max_value=0.99, value=0.8, step=0.05)

        # Calculate required sample size
        analysis = sm_power.TTestPower()
        n = analysis.solve_power(effect_size=effect_size, power=desired_power, alpha=alpha)

        st.subheader("Results")
        st.write(f"**Required sample size**: {np.ceil(n):.0f}")

        # Plot power curve
        sample_sizes = np.arange(5, 200, 5)
        powers = [analysis.solve_power(effect_size=effect_size, nobs1=n, alpha=alpha) for n in sample_sizes]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sample_sizes, y=powers, mode='lines', name='Power Curve',
                                 line=dict(color=colors[0], width=3)))
        fig.add_hline(y=desired_power, line_dash="dash", line_color=colors[1],
                      annotation_text=f"Target Power ({desired_power})")
        fig.update_layout(title="Power Curve",
                          xaxis_title="Sample Size",
                          yaxis_title="Power")
        st.plotly_chart(fig, use_container_width=True)

    elif test_type == "Proportion test":
        col1, col2 = st.columns(2)
        with col1:
            p1 = st.number_input("Proportion in group 1", min_value=0.01, max_value=0.99, value=0.5, step=0.05)
            p2 = st.number_input("Proportion in group 2", min_value=0.01, max_value=0.99, value=0.4, step=0.05)
            alpha = st.number_input("Alpha level", min_value=0.001, max_value=0.2, value=0.05, step=0.01)
        with col2:
            desired_power = st.number_input("Desired power", min_value=0.1, max_value=0.99, value=0.8, step=0.05)
            ratio = st.number_input("Sample size ratio (n2/n1)", min_value=0.1, value=1.0, step=0.1)

        # Calculate required sample size
        effect_size = abs(p1 - p2)
        analysis = sm_power.NormalIndPower()
        n = analysis.solve_power(effect_size=effect_size, power=desired_power, alpha=alpha, ratio=ratio)

        st.subheader("Results")
        st.write(f"**Required sample size per group**: {np.ceil(n):.0f}")
        st.write(f"**Total required sample size**: {np.ceil(n) * (1 + ratio):.0f}")

        # Plot power curve
        sample_sizes = np.arange(5, 200, 5)
        powers = [analysis.solve_power(effect_size=effect_size, nobs1=n, alpha=alpha, ratio=ratio) for n in
                  sample_sizes]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sample_sizes, y=powers, mode='lines', name='Power Curve',
                                 line=dict(color=colors[0], width=3)))
        fig.add_hline(y=desired_power, line_dash="dash", line_color=colors[1],
                      annotation_text=f"Target Power ({desired_power})")
        fig.update_layout(title="Power Curve",
                          xaxis_title="Sample Size per Group",
                          yaxis_title="Power")
        st.plotly_chart(fig, use_container_width=True)


def proportion_tests():
    st.header("Proportion Tests")

    st.markdown("""
    <div class="test-info">
    <h4>About Proportion Tests</h4>
    <p>These tests compare proportions between groups or against a hypothesized value.</p>
    <p><strong>Types of tests:</strong></p>
    <ul>
        <li><strong>One-sample:</strong> Compare sample proportion to expected value</li>
        <li><strong>Two-sample:</strong> Compare proportions between two groups</li>
    </ul>
    <p><strong>Assumptions:</strong> Sufficient sample size (np and n(1-p) > 5), independent observations</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="example-box">
    <h4>Example Use Cases:</h4>
    <ul>
        <li>Testing if survey response rate differs from expected (one-sample)</li>
        <li>Comparing conversion rates between two website designs (two-sample)</li>
        <li>Examining if success rates differ between treatment groups</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    test_type = st.radio(
        "Select proportion test type",
        ["One-sample proportion test", "Two-sample proportion test"]
    )

    colors = get_colors(st.session_state.color_palette, 2)

    if test_type == "One-sample proportion test":
        st.subheader("One-sample Proportion Test")
        col1, col2 = st.columns(2)
        with col1:
            successes = st.number_input("Number of successes", min_value=0, value=50)
            trials = st.number_input("Number of trials", min_value=1, value=100)
        with col2:
            hypothesized = st.number_input("Hypothesized proportion", min_value=0.0, max_value=1.0, value=0.5,
                                           step=0.01)
            alternative = st.selectbox("Alternative hypothesis", ["two-sided", "smaller", "larger"])

        # Perform test
        stat, p_value = proportions_ztest(count=successes, nobs=trials, value=hypothesized, alternative=alternative)
        sample_prop = successes / trials

        st.subheader("Results")
        st.write(f"**Sample proportion**: {sample_prop:.4f}")
        st.write(f"**Hypothesized proportion**: {hypothesized:.4f}")
        st.write(f"**z-statistic**: {stat:.4f}")
        st.write(f"**p-value**: {p_value:.4f}")

        # Plot
        fig = go.Figure()
        fig.add_trace(go.Bar(x=["Sample", "Hypothesized"], y=[sample_prop, hypothesized],
                             marker_color=colors))
        fig.update_layout(title="Proportion Comparison",
                          xaxis_title="",
                          yaxis_title="Proportion")
        st.plotly_chart(fig, use_container_width=True)

    elif test_type == "Two-sample proportion test":
        st.subheader("Two-sample Proportion Test")
        col1, col2 = st.columns(2)
        with col1:
            successes1 = st.number_input("Successes in group 1", min_value=0, value=50)
            trials1 = st.number_input("Trials in group 1", min_value=1, value=100)
        with col2:
            successes2 = st.number_input("Successes in group 2", min_value=0, value=40)
            trials2 = st.number_input("Trials in group 2", min_value=1, value=100)

        alternative = st.selectbox("Alternative hypothesis", ["two-sided", "smaller", "larger"])

        # Perform test
        stat, p_value = proportions_ztest(count=[successes1, successes2], nobs=[trials1, trials2],
                                          alternative=alternative)
        prop1 = successes1 / trials1
        prop2 = successes2 / trials2

        st.subheader("Results")
        st.write(f"**Proportion in group 1**: {prop1:.4f} (n={trials1})")
        st.write(f"**Proportion in group 2**: {prop2:.4f} (n={trials2})")
        st.write(f"**Difference in proportions**: {prop1 - prop2:.4f}")
        st.write(f"**z-statistic**: {stat:.4f}")
        st.write(f"**p-value**: {p_value:.4f}")

        # Plot
        fig = go.Figure()
        fig.add_trace(go.Bar(x=["Group 1", "Group 2"], y=[prop1, prop2],
                             marker_color=colors))
        fig.update_layout(title="Proportion Comparison Between Groups",
                          xaxis_title="",
                          yaxis_title="Proportion")
        st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
