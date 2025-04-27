#!/usr/bin/env python3

"""Database connection configuration"""

# Database configuration
MYSQL_CONFIG = {
    'host': '40.76.125.54',    # Azure VM IP address
    'user': 'shreyaa',         # Username
    'password': 'Shreyaa@123', # Password
    'database': 'shreyaa',     # Database name
    'charset': 'utf8mb4',
    'cursorclass': 'DictCursor'
}