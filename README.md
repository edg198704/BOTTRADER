# Quantum Institutional Bot - Deployment Manual (Windows 10/11 WSL)

This manual guides a non-programmer through deploying the Quantum Trading Bot on Windows using the Windows Subsystem for Linux (WSL) and Docker. This setup ensures the bot runs in a stable, isolated environment identical to institutional servers.

## 1. Prerequisites

### A. Enable WSL2
1. Open **PowerShell** as Administrator.
2. Run: `wsl --install`
3. **Restart your computer** when prompted.
4. After restart, open the **Ubuntu** app (it installs automatically or can be found in the Microsoft Store) to finish the setup (create a username/password).

### B. Install Docker Desktop
1. Download and install **Docker Desktop for Windows** from [docker.com](https://www.docker.com/products/docker-desktop/).
2. Open Docker Desktop Settings (gear icon).
3. Go to **Resources > WSL Integration**.
4. Ensure "Enable integration with my default WSL distro" is checked.
5. Toggle the switch for **Ubuntu** to ON.
6. Click **Apply & Restart**.

## 2. Environment Setup

1. Open your **Ubuntu** terminal.
2. Update your Linux packages to ensure security:
   ```bash
   sudo apt update && sudo apt upgrade -y
   ```
3. Clone the repository (download the bot code):
   ```bash
   git clone https://github.com/edg198704/BOTTRADER.git
   cd BOTTRADER
   ```

## 3. Configuration

1. **Create the Configuration File**:
   Copy the example environment file to a real one.
   ```bash
   cp .env.example .env
   ```

2. **Edit the Configuration**:
   Open the file in the nano editor.
   ```bash
   nano .env
   ```

3. **Fill in your Details**:
   - **Exchange Keys**: If trading live, enter your API Key and Secret. For testing, you can leave them blank or use dummy values if `MockExchange` is selected in `config_enterprise.yaml`.
   - **Telegram**: Enter your Bot Token and Admin Chat ID to receive alerts.
   - **InfluxDB**: Leave the default values for `DOCKER_INFLUXDB_...` unless you are an advanced user.

4. **Save and Exit**:
   - Press `Ctrl+O`, then `Enter` to save.
   - Press `Ctrl+X` to exit.

## 4. Deployment (The Fix)

We have updated the system to handle the installation of complex mathematical libraries automatically.

1. **Build and Start**:
   Run the following command. This will build the 'Brain' of the bot and start the database services.
   ```bash
   docker-compose up -d --build
   ```
   *Note: The first run may take 5-15 minutes as it downloads AI libraries (PyTorch) and compiles optimization tools.*

2. **Verify Status**:
   Check if the containers are running:
   ```bash
   docker-compose ps
   ```
   You should see `quantum_bot`, `quantum_influxdb`, and `quantum_grafana` with status `Up`.

3. **Initial AI Training**:
   The bot needs to initialize its neural networks. Run this command inside the container:
   ```bash
   docker-compose exec trading_bot python train_bot.py
   ```
   *Wait for the success message indicating models have been saved.*

4. **Restart Bot**:
   Apply the new models by restarting the bot service:
   ```bash
   docker-compose restart trading_bot
   ```

## 5. Monitoring & Control

### View Live Logs
To see the bot's decision-making process in real-time:
```bash
docker-compose logs -f trading_bot
```
(Press `Ctrl+C` to exit the log view. The bot keeps running in the background.)

### Access Grafana Dashboard
1. Open your web browser.
2. Go to: [http://localhost:3000](http://localhost:3000)
3. Login with:
   - **User:** `admin`
   - **Password:** `admin` (You will be prompted to change this on first login)
4. Navigate to **Dashboards > Quantum Institutional Dashboard** to view real-time equity, AI confidence, and market regimes.

### Stop the Bot
To stop all services safely:
```bash
docker-compose down
```
