-- MLOps Database Schema Initialization

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Models table
CREATE TABLE IF NOT EXISTS models (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    version VARCHAR(50) NOT NULL,
    model_hash VARCHAR(64),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    stage VARCHAR(50) DEFAULT 'dev',
    metrics JSONB,
    parameters JSONB,
    tags JSONB,
    description TEXT,
    UNIQUE(name, version)
);

CREATE INDEX idx_models_name ON models(name);
CREATE INDEX idx_models_version ON models(version);
CREATE INDEX idx_models_stage ON models(stage);
CREATE INDEX idx_models_created_at ON models(created_at DESC);

-- Deployments table
CREATE TABLE IF NOT EXISTS deployments (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    deployment_id VARCHAR(255) UNIQUE NOT NULL,
    model_id UUID REFERENCES models(id),
    model_name VARCHAR(255) NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    strategy VARCHAR(50) NOT NULL,
    status VARCHAR(50) NOT NULL,
    replicas INTEGER DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    deployed_by VARCHAR(255),
    config JSONB,
    metadata JSONB
);

CREATE INDEX idx_deployments_model_id ON deployments(model_id);
CREATE INDEX idx_deployments_status ON deployments(status);
CREATE INDEX idx_deployments_created_at ON deployments(created_at DESC);

-- Predictions table (for tracking)
CREATE TABLE IF NOT EXISTS predictions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_version VARCHAR(50) NOT NULL,
    prediction JSONB NOT NULL,
    input_data JSONB,
    ground_truth JSONB,
    latency_ms FLOAT,
    error VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_predictions_model_version ON predictions(model_version);
CREATE INDEX idx_predictions_created_at ON predictions(created_at DESC);

-- Data versions table
CREATE TABLE IF NOT EXISTS data_versions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    version_id VARCHAR(255) UNIQUE NOT NULL,
    dataset_name VARCHAR(255) NOT NULL,
    num_rows INTEGER,
    num_features INTEGER,
    data_hash VARCHAR(64),
    storage_path VARCHAR(500),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    parent_version VARCHAR(255),
    metadata JSONB
);

CREATE INDEX idx_data_versions_dataset_name ON data_versions(dataset_name);
CREATE INDEX idx_data_versions_created_at ON data_versions(created_at DESC);

-- Audit logs table
CREATE TABLE IF NOT EXISTS audit_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id VARCHAR(255),
    action VARCHAR(100) NOT NULL,
    resource_type VARCHAR(50),
    resource_id VARCHAR(255),
    status VARCHAR(50),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    ip_address VARCHAR(45),
    details JSONB
);

CREATE INDEX idx_audit_logs_user_id ON audit_logs(user_id);
CREATE INDEX idx_audit_logs_action ON audit_logs(action);
CREATE INDEX idx_audit_logs_timestamp ON audit_logs(timestamp DESC);

-- Metrics table (for monitoring)
CREATE TABLE IF NOT EXISTS metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    metric_name VARCHAR(255) NOT NULL,
    metric_value FLOAT NOT NULL,
    model_version VARCHAR(50),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    tags JSONB
);

CREATE INDEX idx_metrics_name ON metrics(metric_name);
CREATE INDEX idx_metrics_model_version ON metrics(model_version);
CREATE INDEX idx_metrics_timestamp ON metrics(timestamp DESC);

-- Users table (for RBAC)
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    username VARCHAR(255) UNIQUE NOT NULL,
    role VARCHAR(50) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP,
    is_active BOOLEAN DEFAULT true,
    metadata JSONB
);

CREATE INDEX idx_users_username ON users(username);
CREATE INDEX idx_users_role ON users(role);

-- Permissions table
CREATE TABLE IF NOT EXISTS permissions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id),
    resource_type VARCHAR(50) NOT NULL,
    resource_id VARCHAR(255),
    access_level VARCHAR(50) NOT NULL,
    granted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    granted_by VARCHAR(255)
);

CREATE INDEX idx_permissions_user_id ON permissions(user_id);
CREATE INDEX idx_permissions_resource_type ON permissions(resource_type);

-- Drift detection results table
CREATE TABLE IF NOT EXISTS drift_detections (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_version VARCHAR(50) NOT NULL,
    feature_name VARCHAR(255) NOT NULL,
    drift_detected BOOLEAN NOT NULL,
    drift_score FLOAT,
    test_statistic FLOAT,
    p_value FLOAT,
    test_method VARCHAR(50),
    detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_drift_model_version ON drift_detections(model_version);
CREATE INDEX idx_drift_detected_at ON drift_detections(detected_at DESC);

-- A/B test results table
CREATE TABLE IF NOT EXISTS ab_tests (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    test_id VARCHAR(255) UNIQUE NOT NULL,
    control_model VARCHAR(50) NOT NULL,
    treatment_model VARCHAR(50) NOT NULL,
    success_metric VARCHAR(100),
    status VARCHAR(50) NOT NULL,
    start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    end_time TIMESTAMP,
    results JSONB,
    winner VARCHAR(50)
);

CREATE INDEX idx_ab_tests_status ON ab_tests(status);
CREATE INDEX idx_ab_tests_start_time ON ab_tests(start_time DESC);

-- Create views for common queries

-- Active deployments view
CREATE OR REPLACE VIEW active_deployments AS
SELECT
    d.deployment_id,
    d.model_name,
    d.model_version,
    d.strategy,
    d.status,
    d.replicas,
    d.created_at,
    m.metrics as model_metrics
FROM deployments d
LEFT JOIN models m ON d.model_id = m.id
WHERE d.status = 'completed' OR d.status = 'running';

-- Model performance view
CREATE OR REPLACE VIEW model_performance AS
SELECT
    model_version,
    COUNT(*) as total_predictions,
    AVG(latency_ms) as avg_latency_ms,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY latency_ms) as p95_latency_ms,
    PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY latency_ms) as p99_latency_ms,
    COUNT(CASE WHEN error IS NOT NULL THEN 1 END) as error_count,
    COUNT(CASE WHEN error IS NOT NULL THEN 1 END)::FLOAT / COUNT(*) as error_rate
FROM predictions
WHERE created_at > NOW() - INTERVAL '24 hours'
GROUP BY model_version;

-- Grant permissions to mlops user
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO mlops;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO mlops;
GRANT USAGE ON SCHEMA public TO mlops;

-- Insert default admin user
INSERT INTO users (username, role, metadata)
VALUES ('admin', 'admin', '{"created_by": "system"}')
ON CONFLICT (username) DO NOTHING;

-- Success message
DO $$
BEGIN
    RAISE NOTICE 'MLOps database schema initialized successfully!';
    RAISE NOTICE 'Tables created: models, deployments, predictions, data_versions, audit_logs, metrics, users, permissions, drift_detections, ab_tests';
    RAISE NOTICE 'Views created: active_deployments, model_performance';
END $$;
