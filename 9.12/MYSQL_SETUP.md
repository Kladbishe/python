# MySQL Database Setup

## MySQL Installation

### macOS:
```bash
brew install mysql
brew services start mysql
```

### Linux (Ubuntu/Debian):
```bash
sudo apt-get update
sudo apt-get install mysql-server
sudo systemctl start mysql
```

### Windows:
Download and install MySQL from the official website: https://dev.mysql.com/downloads/

## Database Setup

1. Login to MySQL:
```bash
mysql -u root -p
```

2. Create database:
```sql
CREATE DATABASE news_site CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
```

3. Create user (optional):
```sql
CREATE USER 'news_user'@'localhost' IDENTIFIED BY 'your_password';
GRANT ALL PRIVILEGES ON news_site.* TO 'news_user'@'localhost';
FLUSH PRIVILEGES;
```

4. Exit MySQL:
```sql
EXIT;
```

## Configuration Update

Edit the `config.py` file:

```python
# Instead of SQLite:
SQLALCHEMY_DATABASE_URI = 'sqlite:///site.db'

# Use MySQL:
SQLALCHEMY_DATABASE_URI = 'mysql://news_user:your_password@localhost/news_site'
```

Or use environment variable in `.env`:
```
DATABASE_URL=mysql://news_user:your_password@localhost/news_site
```

## MySQL Dependencies Installation

Make sure the required packages are installed:

```bash
pip install Flask-MySQLdb mysqlclient
```

If you encounter issues installing `mysqlclient`, install the required dependencies:

### macOS:
```bash
brew install mysql-client
export PATH="/usr/local/opt/mysql-client/bin:$PATH"
```

### Linux:
```bash
sudo apt-get install python3-dev default-libmysqlclient-dev build-essential
```

## Connection Test

After setup, run the application:

```bash
python app.py
```

Tables will be created automatically on first run.

## Alternative: PyMySQL

If `mysqlclient` doesn't install, you can use PyMySQL:

```bash
pip install PyMySQL
```

And add to the beginning of `app.py`:

```python
import pymysql
pymysql.install_as_MySQLdb()
```

Then use URL:
```
mysql+pymysql://news_user:your_password@localhost/news_site
```
