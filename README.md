# Quantum Institutional Bot - Deployment Manual (Windows 10/11 WSL)

This manual guides a non-programmer through deploying the Quantum Trading Bot on Windows using the Windows Subsystem for Linux (WSL) and Docker.

## 1. Prerequisites

### A. Enable WSL2
1. Open **PowerShell** as Administrator.
2. Run: `wsl --install`
3. Restart your computer when prompted.
4. After restart, open the **Ubuntu** app (it installs automatically or can be found in the Microsoft Store) to finish the setup (create a username/password).

### B. Install Docker Desktop
1. Download and install **Docker Desktop for Windows**.
2. Open Docker Desktop Settings (gear icon).
3. Go to **Resources > WSL Integration**.
4. Ensure "Enable integration with my default WSL distro" is checked.
5. Toggle the switch for **Ubuntu**.
6. Click **Apply & Restart**.

## 2. Environment Setup

1. Open your **Ubuntu** terminal.
2. Update your packages:
   ```bash
   sudo apt update && sudo apt upgrade -y
   ```
3. Clone the repository:
   ```bash
   git clone https://github.com/edg198704/BOTTRADER.git
   cd BOTTRADER
   ```

## 3. Configuration

1. Create the environment file:
   ```bash
   nano .env
   ```
2. Paste the following content (adjust keys if you have them, otherwise leave dummy values for testing):
   ```env
   # Exchange Keys (Required for Live Trading, ignored for Mock)
   BOT_EXCHANGE_API_KEY=your_api_key_here
   BOT_EXCHANGE_API_SECRET=your_api_secret_here
   
   # Telegram (Optional)
   BOT_TELEGRAM_BOT_TOKEN=your_telegram_token_here
   
   # InfluxDB (Do not change unless you know what you are doing)
   DOCKER_INFLUXDB_INIT_USERNAME=admin
   DOCKER_INFLUXDB_INIT_PASSWORD=quantum_secure_password
   DOCKER_INFLUXDB_INIT_ORG=quantum_org
   DOCKER_INFLUXDB_INIT_BUCKET=bot_metrics
   DOCKER_INFLUXDB_INIT_ADMIN_TOKEN=my-super-secret-auth-token
   
   # Bot Internal Config
   INFLUXDB_URL=http://influxdb:8086
   INFLUXDB_TOKEN=my-super-secret-auth-token
   INFLUXDB_ORG=quantum_org
   INFLUXDB_BUCKET=bot_metrics
   ```
3. Press `Ctrl+O`, `Enter` to save, then `Ctrl+X` to exit.

## 4. Deployment

1. Build and start the containers:
   ```bash
   docker-compose up -d --build
   ```
   *This may take a few minutes as it downloads dependencies and compiles the AI engine.*

2. **CRITICAL STEP: Initial AI Training**
   The bot starts with no brain. You must train it once before it can trade.
   ```bash
   docker-compose exec trading_bot python train_bot.py
   ```
   *Wait for this to complete. It will download data and train the Ensemble models.*

3. Restart the bot to load the new models:
   ```bash
   docker-compose restart trading_bot
   ```

## 5. Monitoring & Control

### View Logs
To see what the bot is doing in real-time:
```bash
docker-compose logs -f trading_bot
```
(Press `Ctrl+C` to exit logs)

### Access Grafana Dashboard
1. Open your web browser.
2. Go to: [http://localhost:3000](http://localhost:3000)
3. Login with:
   - **User:** `admin`
   - **Password:** `admin` (You will be asked to change this)
4. Navigate to **Dashboards > Quantum Institutional Dashboard**.

### Stop the Bot
To stop all services:
```bash
docker-compose down
```
