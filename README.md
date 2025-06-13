# PG&E Rate Calculator ⚡

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.45+-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A comprehensive web application to help you find the best PG&E electricity rate plan based on your actual usage patterns. Save money by comparing all major PG&E rate plans with your real data.

![PG&E Calculator Demo](https://via.placeholder.com/800x400/1f77b4/ffffff?text=PG%26E+Rate+Calculator+Demo)

## ✨ Features

### 🔍 **Comprehensive Rate Plan Analysis**
- **Tiered Rate Plan (E-1)**: Traditional tier-based pricing
- **Time-of-Use Plans**: E-TOU-C (with baseline), E-TOU-D (simple TOU)
- **Electric Home Plans**: E-ELEC optimized for all-electric homes
- **Electric Vehicle Plans**: EV2-A (home charging), EV-B (time-of-use for EVs)

### 📊 **Advanced Analytics & Visualizations**
- Interactive hourly usage pattern analysis
- Daily usage trends with customizable date ranges
- Peak vs off-peak usage breakdown
- Cost comparison across all rate plans
- Savings potential analysis with recommendations

### 🎛️ **Smart Configuration**
- Territory-specific baseline allowances (P, Q, R, S, T, V, W, X, Y, Z)
- Heating system type consideration (Basic Electric vs All Electric)
- Baseline allowance calculations for accurate tiered pricing
- Season-aware rate calculations (Summer/Winter)

### 💡 **Enhanced User Experience**
- Modern, responsive web interface
- Real-time cost calculations
- Detailed plan comparisons with rankings
- Interactive charts and graphs
- Mobile-friendly design

## 🚀 Quick Start

### Prerequisites
- Python 3.10 or higher
- [Poetry](https://python-poetry.org/docs/#installation) for dependency management

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/ashaychangwani/pge-calc.git
   cd pge-calc
   ```

2. **Install dependencies**:
   ```bash
   poetry install
   ```

3. **Run the application**:
   ```bash
   # Using the convenient runner script
   python run_app.py
   
   # Or directly with Poetry
   poetry run streamlit run main.py
   
   # Or using the installed command
   poetry run pge-calc
   ```

4. **Open your browser** and navigate to `http://localhost:8501`

## 📖 Usage Guide

### Step 1: Configure Your Settings
- **Territory**: Select your baseline territory (found on your PG&E bill)
- **Heating Type**: Choose between Basic Electric or All Electric

### Step 2: Upload Your Usage Data
1. Visit [PG&E My Account](https://myaccount.pge.com/)
2. Navigate to "Usage & Consumption"
3. Download your usage data in CSV format
4. Upload the file to the calculator

### Step 3: Analyze Your Results
- View your recommended best rate plan
- Explore interactive usage visualizations
- Compare costs across all available plans
- Adjust date ranges to analyze specific periods

## 🏗️ Project Structure

```
pge-calc/
├── src/
│   └── pge_calculator/
│       ├── __init__.py
│       ├── calculator.py      # Core calculation logic
│       └── app.py            # Streamlit web interface
├── data/
│   └── pge_rates.csv         # Current PG&E rate data
├── sample_usage.csv          # Sample usage data for testing
├── main.py                   # Application entry point
├── run_app.py               # Convenient runner script
├── pyproject.toml           # Project configuration
├── README.md               # This file
├── LICENSE                 # MIT License
├── CONTRIBUTING.md         # Contributing guidelines
└── .gitignore             # Git ignore rules
```

## 📊 Understanding Your Territory

Your baseline territory determines your daily allowance for tier 1 pricing. Find your territory on your PG&E bill.

| Territory | Basic Electric (Summer/Winter) | All Electric (Summer/Winter) |
|-----------|-------------------------------|------------------------------|
| **P**     | 13.5 / 11.0 kWh/day          | 15.2 / 26.0 kWh/day         |
| **Q**     | 9.8 / 11.0 kWh/day           | 8.5 / 26.0 kWh/day          |
| **R**     | 17.7 / 10.4 kWh/day          | 19.9 / 26.7 kWh/day         |
| **S**     | 15.0 / 10.2 kWh/day          | 17.8 / 23.7 kWh/day         |
| **T**     | 6.5 / 7.5 kWh/day            | 7.1 / 12.9 kWh/day          |
| **V**     | 7.1 / 8.1 kWh/day            | 10.4 / 19.1 kWh/day         |
| **W**     | 19.2 / 9.8 kWh/day           | 22.4 / 19.0 kWh/day         |
| **X**     | 9.8 / 9.7 kWh/day            | 8.5 / 14.6 kWh/day          |
| **Y**     | 10.5 / 11.1 kWh/day          | 12.0 / 24.0 kWh/day         |
| **Z**     | 5.9 / 7.8 kWh/day            | 6.7 / 15.7 kWh/day          |

**Basic Electric**: No permanently installed electric space heating
**All Electric**: Includes permanently installed electric space heating

## 💰 Rate Plan Details

### Tiered Rate Plan (E-1)
- **Structure**: Fixed rates with tier-based pricing
- **Tier 1**: Below baseline allowance - Lower rate
- **Tier 2**: Above baseline allowance - Higher rate
- **Best for**: Customers with consistent, predictable usage

### Time-of-Use Plans

#### E-TOU-C (Baseline TOU)
- **Peak Hours**: 4-9 PM daily
- **Features**: Different rates for usage above/below baseline
- **Best for**: Customers who can shift usage away from peak hours

#### E-TOU-D (Simple TOU)
- **Peak Hours**: 5-8 PM (summer), 5-8 PM (winter)
- **Features**: No baseline differentiation
- **Best for**: Customers with flexible usage patterns

#### E-ELEC (Electric Home)
- **Peak Hours**: 4-9 PM with super-peak 3-4 PM (summer)
- **Features**: Optimized for all-electric homes
- **Best for**: Customers with electric heating and appliances

### Electric Vehicle Plans

#### EV2-A (Home Charging)
- **Super Off-Peak**: 12 AM - 3 PM (lowest rates)
- **Peak**: 4-9 PM (highest rates)
- **Best for**: EV owners who can charge during off-peak hours

#### EV-B (EV Time-of-Use)
- **Super Off-Peak**: 12 AM - 7 AM
- **Peak**: 2-9 PM
- **Best for**: EV owners with flexible charging schedules

## 🛠️ Development

### Setup Development Environment

```bash
# Clone and install
git clone https://github.com/ashaychangwani/pge-calc.git
cd pge-calc
poetry install --with dev

# Install pre-commit hooks
poetry run pre-commit install

# Run the application
python run_app.py
```

### Running Tests

```bash
# Run tests (when available)
poetry run pytest

# Type checking
poetry run mypy src/

# Code formatting
poetry run black src/
poetry run flake8 src/
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details on:

- Setting up the development environment
- Code style and standards
- Submitting pull requests
- Reporting bugs and requesting features

### Quick Contribution Steps

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes and commit: `git commit -m 'Add amazing feature'`
4. Push to the branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

## 📈 Roadmap

- [ ] **Export Features**: Save analysis results as PDF/Excel
- [ ] **Historical Analysis**: Compare usage across multiple months
- [ ] **Cost Predictions**: Forecast future costs based on usage patterns
- [ ] **Smart Recommendations**: AI-powered usage optimization suggestions
- [ ] **Mobile App**: Native mobile application
- [ ] **API Integration**: Real-time rate updates from PG&E
- [ ] **Multi-Utility Support**: Support for other California utilities

## ❓ FAQ

**Q: How often are rates updated?**
A: Rate data is manually updated when PG&E publishes new tariffs. Check the repository for the latest updates.

**Q: Is my usage data stored or shared?**
A: No. All processing happens locally in your browser. No data is transmitted or stored on external servers.

**Q: Can I use this for other utilities?**
A: Currently, this calculator is specifically designed for PG&E rates. Support for other utilities may be added in the future.

**Q: What if my bill shows different rates?**
A: This calculator provides estimates based on published rates. Actual bills may vary due to additional fees, taxes, and other charges not included in the calculation.

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/ashaychangwani/pge-calc/issues)
- **Discussions**: [GitHub Discussions](https://github.com/ashaychangwani/pge-calc/discussions)
- **Email**: ashaychangwani@gmail.com

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **PG&E** for providing public rate information
- **Streamlit** for the excellent web framework
- **Plotly** for interactive visualizations
- **Poetry** for modern Python dependency management

---

<div align="center">
  <strong>Save money on your electricity bill with data-driven decisions! ⚡💡</strong>
</div> 