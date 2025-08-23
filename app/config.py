"""
Configuration management for MAILMIND2.0
Handles application settings, environment variables, and configuration validation
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from dotenv import load_dotenv

# ✅ Load .env automatically
load_dotenv()


@dataclass
class EmailConfig:
    """Email server configuration"""
    smtp_server: str = ""
    smtp_port: int = 587
    imap_server: str = ""
    imap_port: int = 993
    username: str = ""
    password: str = ""
    use_tls: bool = True
    use_ssl: bool = False
    timeout: int = 30


@dataclass
class DatabaseConfig:
    """Database configuration (MySQL version)"""
    host: str = "localhost"
    port: int = 3306
    database: str = "mailmind_db"
    username: str = ""
    password: str = ""
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30

    @property
    def url(self) -> str:
        """Generate MySQL database URL"""
        return f"mysql+pymysql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"


@dataclass
class SecurityConfig:
    """Security and encryption settings"""
    secret_key: str = ""
    encryption_key: str = ""
    jwt_secret: str = ""
    jwt_expiry_hours: int = 24
    password_salt: str = ""
    max_login_attempts: int = 5
    lockout_duration_minutes: int = 30


@dataclass
class ProcessingConfig:
    """Email processing configuration"""
    batch_size: int = 100
    max_concurrent_jobs: int = 5
    retry_attempts: int = 3
    retry_delay_seconds: int = 60
    cleanup_days: int = 30
    auto_archive_days: int = 90
    spam_threshold: float = 0.7
    enable_ai_classification: bool = True
    processing_timeout: int = 300


@dataclass
class ApiConfig:
    """API configuration"""
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    rate_limit_per_minute: int = 100
    max_request_size: int = 10 * 1024 * 1024  # 10MB
    enable_docs: bool = True


@dataclass
class CacheConfig:
    """Cache configuration"""
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None
    default_ttl: int = 3600  # 1 hour
    max_memory: str = "256mb"
    enable_redis: bool = True


@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: str = "logs/mailmind.log"
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    enable_console: bool = True
    enable_file: bool = True


@dataclass
class Config:
    """Main configuration class"""
    environment: str = "development"
    debug: bool = True

    # Component configurations
    email: EmailConfig = field(default_factory=EmailConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    api: ApiConfig = field(default_factory=ApiConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    # Additional settings
    app_name: str = "MAILMIND2.0"
    version: str = "2.0.0"
    timezone: str = "UTC"
    temp_dir: str = "/tmp/mailmind"
    data_dir: str = "data"

    def __post_init__(self):
        """Post-initialization setup (skip validation here)"""
        self._create_directories()

    def _validate_config(self):
        """Validate configuration values"""
        errors = []

        # Validate email configuration
        if not self.email.smtp_server:
            errors.append("SMTP server is required")
        if not self.email.username:
            errors.append("Email username is required")

        # Validate database configuration
        if not self.database.username:
            errors.append("Database username is required")
        if not self.database.password:
            errors.append("Database password is required")

        # Validate security configuration
        if not self.security.secret_key:
            errors.append("Secret key is required")

        # Validate API port
        if not (1 <= self.api.port <= 65535):
            errors.append("API port must be between 1 and 65535")

        if errors:
            raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")

    def _create_directories(self):
        """Create necessary directories"""
        directories = [
            self.temp_dir,
            self.data_dir,
            os.path.dirname(self.logging.file_path)
        ]
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_env(cls) -> 'Config':
        """Create configuration from environment variables (validated after population)"""
        # Create instance without triggering __post_init__ validation
        config = cls.__new__(cls)
        super(Config, config).__init__()

        # Environment
        config.environment = os.getenv('MAILMIND_ENV', 'development')
        config.debug = os.getenv('MAILMIND_DEBUG', 'true').lower() == 'true'

        # Email configuration
        config.email = EmailConfig(
            smtp_server=os.getenv('SMTP_SERVER', ''),
            smtp_port=int(os.getenv('SMTP_PORT', '587')),
            imap_server=os.getenv('IMAP_SERVER', ''),
            imap_port=int(os.getenv('IMAP_PORT', '993')),
            username=os.getenv('EMAIL_USERNAME', ''),
            password=os.getenv('EMAIL_PASSWORD', ''),
            use_tls=os.getenv('EMAIL_USE_TLS', 'true').lower() == 'true',
            use_ssl=os.getenv('EMAIL_USE_SSL', 'false').lower() == 'true',
        )

        # Database configuration (MySQL defaults)
        config.database = DatabaseConfig(
            host=os.getenv('DB_HOST', 'localhost'),
            port=int(os.getenv('DB_PORT', '3306')),
            database=os.getenv('DB_NAME', 'mailmind_db'),
            username=os.getenv('DB_USERNAME', ''),
            password=os.getenv('DB_PASSWORD', ''),
        )

        # Security configuration
        config.security = SecurityConfig(
            secret_key=os.getenv('SECRET_KEY', ''),
            encryption_key=os.getenv('ENCRYPTION_KEY', ''),
            jwt_secret=os.getenv('JWT_SECRET', ''),
        )

        # API configuration
        config.api = ApiConfig(
            host=os.getenv('API_HOST', '0.0.0.0'),
            port=int(os.getenv('API_PORT', '8000')),
            debug=os.getenv('API_DEBUG', str(config.debug)).lower() == 'true',
        )

        # Cache configuration
        config.cache = CacheConfig(
            redis_host=os.getenv('REDIS_HOST', 'localhost'),
            redis_port=int(os.getenv('REDIS_PORT', '6379')),
            redis_password=os.getenv('REDIS_PASSWORD'),
            redis_db=int(os.getenv('REDIS_DB', '0')),
        )

        # Processing configuration
        config.processing = ProcessingConfig(
            batch_size=int(os.getenv('PROCESSING_BATCH_SIZE', '100')),
            max_concurrent_jobs=int(os.getenv('MAX_CONCURRENT_JOBS', '5')),
        )

        # Logging configuration
        config.logging = LoggingConfig(
            level=os.getenv('LOG_LEVEL', 'INFO'),
            file_path=os.getenv('LOG_FILE_PATH', 'logs/mailmind.log')
        )

        # ✅ Validate AFTER population
        config._validate_config()
        config._create_directories()

        return config

    @classmethod
    def from_file(cls, config_path: str) -> 'Config':
        """Load configuration from JSON file"""
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_file, 'r') as f:
            config_data = json.load(f)

        return cls.from_dict(config_data)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """Create configuration from dictionary"""
        config = cls()
        def update_config(obj, data):
            for key, value in data.items():
                if hasattr(obj, key):
                    attr = getattr(obj, key)
                    if hasattr(attr, '__dict__'):
                        update_config(attr, value)
                    else:
                        setattr(obj, key, value)
        update_config(config, config_dict)
        return config

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        def config_to_dict(obj):
            if hasattr(obj, '__dict__'):
                result = {}
                for key, value in obj.__dict__.items():
                    if not key.startswith('_'):
                        if hasattr(value, '__dict__'):
                            result[key] = config_to_dict(value)
                        else:
                            result[key] = value
                return result
            return obj
        return config_to_dict(self)

    def save_to_file(self, config_path: str):
        """Save configuration to JSON file"""
        config_file = Path(config_path)
        config_file.parent.mkdir(parents=True, exist_ok=True)
        with open(config_file, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    def is_production(self) -> bool:
        return self.environment.lower() == 'production'

    def is_development(self) -> bool:
        return self.environment.lower() == 'development'


# Global configuration instance
_config: Optional[Config] = None

def get_config() -> Config:
    """Get the global configuration instance"""
    global _config
    if _config is None:
        config_file = os.getenv('MAILMIND_CONFIG_FILE', 'config.json')
        if os.path.exists(config_file):
            _config = Config.from_file(config_file)
        else:
            _config = Config.from_env()
    return _config

def set_config(config: Config):
    """Set the global configuration instance"""
    global _config
    _config = config

def reload_config():
    """Reload configuration from source"""
    global _config
    _config = None
    return get_config()
