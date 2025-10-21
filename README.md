<div align="center">
  <img src="assets/BesiegeField_logo.png" alt="BesiegeField Logo" width="50%">
</div>

<div align="center">

**The First Framework for LLM-Driven Machine Design in Besiege**

Paper: [Agentic Design of Compositional Machines](https://arxiv.org/abs/2510.14980)

<a href="https://arxiv.org/abs/2510.14980"><img src="https://img.shields.io/badge/arXiv-2510.14980-b31b1b?style=flat-square&logo=arxiv&logoColor=white"></a>
<a href="https://besiegefield.github.io/"><img src="https://img.shields.io/badge/Project-Website-42a5f5?style=flat-square&logo=firefox&logoColor=white"></a>

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8+-green.svg)](https://www.python.org/)
[![Besiege](https://img.shields.io/badge/besiege-v1.60--22044-orange.svg)](https://store.steampowered.com/app/346010/Besiege)
[![Ubuntu](https://img.shields.io/badge/ubuntu-22.04-purple.svg)](https://ubuntu.com/)

[Installation](#-installation) • [Quick Start](#-quick-start) • [Training](#-llm-fine-tuning) • [Citation](#-citation)

</div>



<p align="center">
  <img src="https://github.com/Godheritage/BesiegeField/blob/main/assets/throwing.gif?raw=true" alt="Demo animation" width="800">
</p>

---
## 📋 Table of Contents


- [Overview](#-overview)
- [Installation](#-installation)
  - [Besiege Environment Setup](#1-besiege-environment-setup)
  - [AgenticFlow Installation](#2-agenticflow-installation)
- [Quick Start](#-quick-start)
- [Fine-tuning](#-llm-fine-tuning)
- [Performance Leaderboard](#-performance-leaderboard)
- [RL Fine-tuning Results](#-rl-finetuned-llm-results)
- [License](#-license)
- [Acknowledgement](#-acknowledgement)
- [Citation](#-citation)

---

## 🌟 Overview

BesiegeField is a cutting-edge framework that enables Large Language Models (LLMs) to autonomously design and build complex machines in the Besiege physics-based game environment. This project bridges AI reasoning with creative engineering tasks.

---

## 🚀 Installation

### 1. Besiege Environment Setup

#### 📦 System Requirements

| Component | Version |
|-----------|---------|
| **Besiege** | Linux v1.60-22044 |
| **Ubuntu** | 22.04 |
| **GLIBC** | 2.33 – 2.35 |
| **Mono** | ≥ 6.8.0.105 |

#### 🎯 Obtain the Game

**Step 1:** Purchase the official copy on [Steam](https://store.steampowered.com/app/346010/Besiege)

**Step 2:** Download [DepotDownloader](https://github.com/SteamRE/DepotDownloader)

**Step 3:** Download Besiege v1.60-22044

```bash
./DepotDownloader -app 346010 -depot 346016 -manifest 2732248020700221971 \
  -username <steam_user> -password <password>
```

**Step 4:** Download v1.20-17395 executables (required for headless operation)

```bash
./DepotDownloader -app 346010 -depot 346016 -manifest 5506301120812842666 \
  -username <steam_user> -password <password>
```

> 💡 **Tip:** Find other manifests on [SteamDB](https://steamdb.info/depot/346016/manifests) if needed.

#### 🔌 Download the Plugin

📥 [BesiegeField Plugin (Google Drive)](https://drive.google.com/file/d/1NPjb1urndwF7zWjV66B8rjmXGT-SrcFj/view?usp=sharing)

#### 🛠️ Install Dependencies

**Standard Installation:**
```bash
sudo apt install mono-complete xvfb  # xvfb only for headless workstation
mono --version  # Verify ≥ 6.8.0.105
```

<details>
<summary>📦 <b>Offline/Manual Installation</b> (click to expand)</summary>

If `apt` is unavailable, use manual installation:

```bash
# Install mono
cd /path/to/tar
tar -xzf mono-complete-offline.tar.gz
for deb in *.deb; do dpkg -x "$deb" .; done

export PATH="/path/to/mono/usr/bin:$PATH"
export LD_LIBRARY_PATH="/path/to/mono/usr/lib:$LD_LIBRARY_PATH"
export PKG_CONFIG_PATH="/path/to/mono/usr/lib/pkgconfig:$PKG_CONFIG_PATH"

# Make permanent
cat >> ~/.bashrc <<EOF
export PATH="/path/to/mono/usr/bin:\$PATH"
export LD_LIBRARY_PATH="/path/to/mono/usr/lib:\$LD_LIBRARY_PATH"
export PKG_CONFIG_PATH="/path/to/mono/usr/lib/pkgconfig:\$PKG_CONFIG_PATH"
EOF
source ~/.bashrc

# Install xvfb
cd /path/to/xvfb
tar -xzf xvfb-offline.tar.gz
dpkg -i *.deb
```

</details>

#### ⚙️ Install BesiegeField Plugin

**Step 1:** Extract the plugin archive and **copy all files** into the v1.60-22044 game folder

**Step 2:** Copy **Besiege.x86** & **Besiege.x86_64** from v1.20-17395 into v1.60-22044, **overwriting** the originals

> ⚠️ **Warning:** This enables headless/code control but makes normal GUI start unstable. Keep a backup if you want to launch v1.60 visually.

**Step 3:** Set permissions

```bash
chmod -R 777 /path/to/Besiege
```

**Step 4:** Test the vanilla game (use backup copy)

```bash
cd /path/to/backup/Besiege && ./run.sh
```

---

### 2. AgenticFlow Installation

#### 🐍 Create Conda Environment

```bash
conda env create -f environment_inferenceonly.yaml
conda activate <env_name>
```

#### 📂 Path Configuration

**Folder Structure:**
```
your-project/
├── Besiege/                  # Game installation
└── AgenticCodes/             # Framework code
```

**Edit `AgenticCodes/config.py`:**

| Parameter | Description |
|-----------|-------------|
| `APIPATH` | Path to file storing LLM type, API key, etc. **Fill it in yourself.** |
| `DEFAULT_SAVE_ROOT` | Root directory for LLM outputs |
| `SCRIPT_PATH` | Must point to `Besiege/run_besiegefield.sh` |

---

## 🎯 Quick Start

### 🏹 Catapult Task

Design a machine to throw projectiles:

```bash
python main.py \
  -use_model deepseek-chat \
  -task catapult/catapult_level1 \
  -env_num 2 \
  -user_input "Design a machine to throw a boulder (type id 36) in a parabolic trajectory."
```

### 🚗 Car Task

Design a machine to move forward:

```bash
python main.py \
  -use_model deepseek-chat \
  -task car/car_level1 \
  -env_num 2 \
  -user_input "Design a machine to move forward on a straight road."
```

### 📝 Available Tasks

Explore all available tasks in `environments/env_files/level_menus.json`

### 🎮 Testing Your Designs

1. Generated `.bsg` machine files appear in `DEFAULT_SAVE_ROOT`
2. Copy them to `Besiege/Besiege_Data/SavedMachines`
3. Run `./run.sh` to launch the game
4. Inspect and test your AI-designed machines in-game!

---

## 🔧 LLM Fine-tuning

### 📦 Install Training Environment

Add training-related packages:

```bash
conda activate <env_name>
pip install -r requirements_rl.txt
```

---

### ❄️ Cold Start Training

#### Step 1: Download Dataset

```bash
cd PostTraining/ColdStart/dataset
./download_dataset.sh
```

#### Step 2: Run Cold Start

```bash
cd ..
./run_cold_start.sh
```

#### Step 3: Merge Checkpoints

Fill the paths in `merge_ckpts.py` before running:

```bash
python merge_ckpts.py
```

---

### 🎓 Reinforcement Learning

Configure `rl_config.yaml` with your settings, then run:

```bash
cd PostTraining/RL
./rl_single_agent_light.sh
```

---

## 📊 Performance Leaderboard

### 🎯 Catapult Task

Performance metrics across different models and methods:

<table>
<thead>
<tr>
<th rowspan="2" align="left">Models</th>
<th colspan="3" align="center"><b>Single-agent</b></th>
<th colspan="3" align="center"><b>Iterative Editing</b></th>
<th colspan="3" align="center"><b>Hierarchical Design</b></th>
</tr>
<tr>
<th align="center"><b>Mean</b></th>
<th align="center"><b>Max</b></th>
<th align="center"><b>Std</b></th>
<th align="center"><b>Mean</b></th>
<th align="center"><b>Max</b></th>
<th align="center"><b>Std</b></th>
<th align="center"><b>Mean</b></th>
<th align="center"><b>Max</b></th>
<th align="center"><b>Std</b></th>
</tr>
</thead>
<tbody>
<tr>
<td><b>Gemini 2.5 Pro</b></td>
<td align="center">2.30</td>
<td align="center">9.00</td>
<td align="center">3.86</td>
<td align="center">4.67</td>
<td align="center"><b>21.95</b></td>
<td align="center">8.68</td>
<td align="center"><b>9.83</b></td>
<td align="center"><b>18.19</b></td>
<td align="center">8.35</td>
</tr>
<tr>
<td><b>OpenAI o3</b></td>
<td align="center">2.87</td>
<td align="center">5.22</td>
<td align="center">1.96</td>
<td align="center"><b>9.14</b></td>
<td align="center">14.01</td>
<td align="center">3.71</td>
<td align="center">2.00</td>
<td align="center">11.11</td>
<td align="center">3.98</td>
</tr>
<tr>
<td><b>Qwen3-Coder-480B-A35B</b></td>
<td align="center">1.75</td>
<td align="center">9.24</td>
<td align="center">3.17</td>
<td align="center">5.10</td>
<td align="center">12.02</td>
<td align="center">5.54</td>
<td align="center">3.90</td>
<td align="center">6.52</td>
<td align="center">2.54</td>
</tr>
<tr>
<td><b>Doubao Seed 1.6-250615</b></td>
<td align="center">3.18</td>
<td align="center">8.20</td>
<td align="center">2.99</td>
<td align="center">4.82</td>
<td align="center">9.10</td>
<td align="center">3.41</td>
<td align="center">1.73</td>
<td align="center">4.76</td>
<td align="center">2.39</td>
</tr>
<tr>
<td><b>Claude Opus 4-20250514</b></td>
<td align="center">1.19</td>
<td align="center">4.82</td>
<td align="center">2.21</td>
<td align="center">1.18</td>
<td align="center">4.91</td>
<td align="center">2.18</td>
<td align="center">2.27</td>
<td align="center">9.32</td>
<td align="center">4.22</td>
</tr>
<tr>
<td><b>DeepSeek-V3</b></td>
<td align="center"><b>3.50</b></td>
<td align="center">4.86</td>
<td align="center">2.17</td>
<td align="center">3.07</td>
<td align="center">5.24</td>
<td align="center">2.55</td>
<td align="center">2.41</td>
<td align="center">4.93</td>
<td align="center">2.58</td>
</tr>
<tr>
<td><b>Kimi K2-0711-preview</b></td>
<td align="center">2.57</td>
<td align="center"><b>9.05</b></td>
<td align="center">3.72</td>
<td align="center">2.82</td>
<td align="center">11.39</td>
<td align="center">5.23</td>
<td align="center">5.39</td>
<td align="center">12.02</td>
<td align="center">5.16</td>
</tr>
<tr>
<td><b>Llama 4 Scout 17B 16E</b></td>
<td align="center">3.18</td>
<td align="center">5.64</td>
<td align="center">1.95</td>
<td align="center">1.28</td>
<td align="center">5.94</td>
<td align="center">2.41</td>
<td align="center">3.59</td>
<td align="center">11.83</td>
<td align="center">4.15</td>
</tr>
</tbody>
</table>

### 🚗 Car Task

Performance metrics across different models and methods:

<table>
<thead>
<tr>
<th rowspan="2" align="left">Models</th>
<th colspan="3" align="center"><b>Single-agent</b></th>
<th colspan="3" align="center"><b>Iterative Editing</b></th>
<th colspan="3" align="center"><b>Hierarchical Design</b></th>
</tr>
<tr>
<th align="center"><b>Mean</b></th>
<th align="center"><b>Max</b></th>
<th align="center"><b>Std</b></th>
<th align="center"><b>Mean</b></th>
<th align="center"><b>Max</b></th>
<th align="center"><b>Std</b></th>
<th align="center"><b>Mean</b></th>
<th align="center"><b>Max</b></th>
<th align="center"><b>Std</b></th>
</tr>
</thead>
<tbody>
<tr>
<td><b>Gemini 2.5 Pro</b></td>
<td align="center"><b>33.96</b></td>
<td align="center"><b>40.85</b></td>
<td align="center">6.73</td>
<td align="center"><b>34.34</b></td>
<td align="center"><b>41.66</b></td>
<td align="center">13.96</td>
<td align="center"><b>29.96</b></td>
<td align="center"><b>41.52</b></td>
<td align="center">7.78</td>
</tr>
<tr>
<td><b>OpenAI o3</b></td>
<td align="center">15.28</td>
<td align="center">32.08</td>
<td align="center">8.97</td>
<td align="center">14.34</td>
<td align="center">35.08</td>
<td align="center">11.79</td>
<td align="center">28.39</td>
<td align="center">36.18</td>
<td align="center">11.01</td>
</tr>
<tr>
<td><b>Qwen3-Coder-480B-A35B</b></td>
<td align="center">8.87</td>
<td align="center">11.50</td>
<td align="center">4.46</td>
<td align="center">15.24</td>
<td align="center">28.95</td>
<td align="center">13.12</td>
<td align="center">12.59</td>
<td align="center">34.05</td>
<td align="center">10.78</td>
</tr>
<tr>
<td><b>Doubao Seed 1.6-250615</b></td>
<td align="center">3.51</td>
<td align="center">9.40</td>
<td align="center">4.85</td>
<td align="center">8.11</td>
<td align="center">10.04</td>
<td align="center">3.58</td>
<td align="center">18.75</td>
<td align="center">26.02</td>
<td align="center">4.38</td>
</tr>
<tr>
<td><b>Claude Opus 4-20250514</b></td>
<td align="center">9.83</td>
<td align="center">12.98</td>
<td align="center">1.28</td>
<td align="center">8.07</td>
<td align="center">28.04</td>
<td align="center">12.48</td>
<td align="center">14.56</td>
<td align="center">38.67</td>
<td align="center">20.69</td>
</tr>
<tr>
<td><b>DeepSeek-V3</b></td>
<td align="center">9.06</td>
<td align="center">10.53</td>
<td align="center">3.68</td>
<td align="center">8.23</td>
<td align="center">18.84</td>
<td align="center">7.12</td>
<td align="center">17.92</td>
<td align="center">31.94</td>
<td align="center">12.85</td>
</tr>
<tr>
<td><b>Kimi K2-0711-preview</b></td>
<td align="center">1.75</td>
<td align="center">8.09</td>
<td align="center">2.80</td>
<td align="center">14.36</td>
<td align="center">28.34</td>
<td align="center">9.47</td>
<td align="center">1.94</td>
<td align="center">14.99</td>
<td align="center">5.48</td>
</tr>
<tr>
<td><b>Llama 4 Scout 17B 16E</b></td>
<td align="center">0.02</td>
<td align="center">0.03</td>
<td align="center">0.01</td>
<td align="center">3.04</td>
<td align="center">12.76</td>
<td align="center">5.23</td>
<td align="center">1.55</td>
<td align="center">2.00</td>
<td align="center">0.32</td>
</tr>
</tbody>
</table>

---

## 🎓 RL-Finetuned LLM Results

Performance comparison of Qwen2.5-14B-Instruct model with different training strategies:

<table>
<thead>
<tr>
<th rowspan="2" align="left">Models</th>
<th colspan="3" align="center"><b>Catapult</b></th>
<th colspan="3" align="center"><b>Car</b></th>
</tr>
<tr>
<th align="center"><b>Validity Ratio</b></th>
<th align="center"><b>Mean Score</b></th>
<th align="center"><b>Max Score</b></th>
<th align="center"><b>Validity Ratio</b></th>
<th align="center"><b>Mean Score</b></th>
<th align="center"><b>Max Score</b></th>
</tr>
</thead>
<tbody>
<tr>
<td><b>Qwen2.5-14B-Instruct</b></td>
<td align="center">11/50</td>
<td align="center">0.06</td>
<td align="center">2.41</td>
<td align="center">46/50</td>
<td align="center">4.97</td>
<td align="center">19.10</td>
</tr>
<tr>
<td><b>Qwen2.5-14B-Instruct + Cold-Start</b></td>
<td align="center">9/50</td>
<td align="center">0.11</td>
<td align="center">5.54</td>
<td align="center">40/50</td>
<td align="center">4.67</td>
<td align="center">20.23</td>
</tr>
<tr>
<td><b>Qwen2.5-14B-Instruct + RL</b></td>
<td align="center">12/50</td>
<td align="center">0.13</td>
<td align="center">5.92</td>
<td align="center">41/50</td>
<td align="center">3.72</td>
<td align="center">24.08</td>
</tr>
<tr>
<td><b>Qwen2.5-14B-Instruct + Cold-Start + RL</b></td>
<td align="center">11/50</td>
<td align="center"><b>0.14</b></td>
<td align="center"><b>7.14</b></td>
<td align="center"><b>42/50</b></td>
<td align="center"><b>5.05</b></td>
<td align="center"><b>45.72</b></td>
</tr>
</tbody>
</table>


---

## 📚 Citation

If you find this repository useful for your research or projects, please consider citing our work:

```bibtex
@article{zhang2025besiegefield,
  title={Agentic Design of Compositional Machines},
  author={Zhang, Wenqian and Liu, Weiyang and Liu, Zhen},
  journal={arXiv preprint arXiv:2510.14980},
  year={2025}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---


## 👏 Acknowledgement

We’d like to thank the developers of Besiege for creating such an inspiring game and for nurturing such a vibrant player community — without them, this project wouldn’t exist.

Big thanks also to the BepInEx team for their amazing modding framework, which made it possible for us to push the boundaries of what’s possible in Besiege.

---
## ⭐ Star History

If you find this project helpful, please consider giving it a star! ⭐

---


