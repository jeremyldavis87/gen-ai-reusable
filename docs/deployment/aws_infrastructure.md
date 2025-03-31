# AWS Infrastructure Guide

## Overview

This document describes the AWS infrastructure setup for the Gen AI Services platform. The infrastructure is managed using Infrastructure as Code (IaC) with Terraform.

## Architecture

### Network Architecture

1. VPC Configuration
   ```hcl
   module "vpc" {
     source = "terraform-aws-modules/vpc/aws"
     
     name = "gen-ai-vpc"
     cidr = "10.0.0.0/16"
     
     azs             = ["us-east-1a", "us-east-1b", "us-east-1c"]
     private_subnets = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
     public_subnets  = ["10.0.101.0/24", "10.0.102.0/24", "10.0.103.0/24"]
     
     enable_nat_gateway = true
     single_nat_gateway = true
     
     tags = {
       Environment = "production"
       Project     = "gen-ai-services"
     }
   }
   ```

2. Security Groups
   ```hcl
   resource "aws_security_group" "service" {
     name        = "gen-ai-service-sg"
     description = "Security group for Gen AI services"
     vpc_id      = module.vpc.vpc_id
     
     ingress {
       from_port   = 8000
       to_port     = 8009
       protocol    = "tcp"
       cidr_blocks = ["0.0.0.0/0"]
     }
     
     egress {
       from_port   = 0
       to_port     = 0
       protocol    = "-1"
       cidr_blocks = ["0.0.0.0/0"]
     }
     
     tags = {
       Name = "gen-ai-service-sg"
     }
   }
   ```

### Container Infrastructure

1. ECS Cluster
   ```hcl
   resource "aws_ecs_cluster" "main" {
     name = "gen-ai-cluster"
     
     setting {
       name  = "containerInsights"
       value = "enabled"
     }
     
     tags = {
       Environment = "production"
       Project     = "gen-ai-services"
     }
   }
   ```

2. ECR Repositories
   ```hcl
   resource "aws_ecr_repository" "services" {
     for_each = toset([
       "format-conversion",
       "classification",
       "workflow",
       "search",
       "quality",
       "personalization",
       "document-extraction",
       "conversational",
       "code"
     ])
     
     name = "gen-ai-${each.key}"
     
     image_scanning_configuration {
       scan_on_push = true
     }
     
     tags = {
       Environment = "production"
       Project     = "gen-ai-services"
     }
   }
   ```

### Database Infrastructure

1. RDS Instance
   ```hcl
   resource "aws_db_instance" "main" {
     identifier           = "gen-ai-db"
     engine              = "postgres"
     engine_version      = "13.7"
     instance_class      = "db.t3.micro"
     allocated_storage   = 20
     storage_type        = "gp2"
     
     db_name             = "genai"
     username           = "genai_admin"
     password           = var.db_password
     
     vpc_security_group_ids = [aws_security_group.db.id]
     db_subnet_group_name   = aws_db_subnet_group.main.name
     
     backup_retention_period = 7
     backup_window          = "03:00-04:00"
     
     tags = {
       Environment = "production"
       Project     = "gen-ai-services"
     }
   }
   ```

2. ElastiCache
   ```hcl
   resource "aws_elasticache_cluster" "redis" {
     cluster_id           = "gen-ai-cache"
     engine              = "redis"
     node_type           = "cache.t3.micro"
     num_cache_nodes     = 1
     parameter_group_family = "redis6.x"
     port                = 6379
     
     security_group_ids  = [aws_security_group.cache.id]
     subnet_group_name   = aws_elasticache_subnet_group.main.name
     
     tags = {
       Environment = "production"
       Project     = "gen-ai-services"
     }
   }
   ```

### Storage Infrastructure

1. S3 Buckets
   ```hcl
   resource "aws_s3_bucket" "documents" {
     bucket = "gen-ai-documents"
     
     versioning {
       enabled = true
     }
     
     server_side_encryption_configuration {
       rule {
         apply_server_side_encryption_by_default {
           sse_algorithm = "AES256"
         }
       }
     }
     
     tags = {
       Environment = "production"
       Project     = "gen-ai-services"
     }
   }
   ```

### Monitoring Infrastructure

1. CloudWatch
   ```hcl
   resource "aws_cloudwatch_log_group" "services" {
     for_each = toset([
       "format-conversion",
       "classification",
       "workflow",
       "search",
       "quality",
       "personalization",
       "document-extraction",
       "conversational",
       "code"
     ])
     
     name              = "/ecs/gen-ai-${each.key}"
     retention_in_days = 30
     
     tags = {
       Environment = "production"
       Project     = "gen-ai-services"
     }
   }
   ```

2. X-Ray
   ```hcl
   resource "aws_xray_sampling_rule" "main" {
     rule_name      = "gen-ai-sampling"
     priority       = 1000
     version        = 1
     reservoir_size = 1
     fixed_rate     = 0.05
     
     host = "*"
     http_method = "*"
     url_path = "*"
     service_name = "*"
     service_type = "*"
     
     resource_arn = "*"
     
     attributes = {
       Environment = "production"
       Project     = "gen-ai-services"
     }
   }
   ```

### Security Infrastructure

1. IAM Roles
   ```hcl
   resource "aws_iam_role" "ecs_task_execution" {
     name = "gen-ai-ecs-task-execution"
     
     assume_role_policy = jsonencode({
       Version = "2012-10-17"
       Statement = [
         {
           Action = "sts:AssumeRole"
           Effect = "Allow"
           Principal = {
             Service = "ecs-tasks.amazonaws.com"
           }
         }
       ]
     })
     
     tags = {
       Environment = "production"
       Project     = "gen-ai-services"
     }
   }
   ```

2. Secrets Manager
   ```hcl
   resource "aws_secretsmanager_secret" "api_keys" {
     name = "gen-ai-api-keys"
     
     tags = {
       Environment = "production"
       Project     = "gen-ai-services"
     }
   }
   ```

## Deployment Process

### Infrastructure Deployment

1. Initialize Terraform
   ```bash
   cd deploy/infrastructure
   terraform init
   ```

2. Plan Changes
   ```bash
   terraform plan -out=tfplan
   ```

3. Apply Changes
   ```bash
   terraform apply tfplan
   ```

### Service Deployment

1. Build Images
   ```bash
   ./scripts/build_images.sh
   ```

2. Push Images
   ```bash
   ./scripts/push_images.sh
   ```

3. Deploy Services
   ```bash
   ./scripts/deploy_services.sh
   ```

## Monitoring and Maintenance

### CloudWatch Dashboards

1. Service Metrics
   - CPU Utilization
   - Memory Usage
   - Request Count
   - Error Rate
   - Response Time

2. Infrastructure Metrics
   - Database Connections
   - Cache Hit Rate
   - Storage Usage
   - Network Traffic

### Alerts

1. Service Alerts
   ```hcl
   resource "aws_cloudwatch_metric_alarm" "service_health" {
     alarm_name          = "gen-ai-service-health"
     comparison_operator = "GreaterThanThreshold"
     evaluation_periods  = "2"
     metric_name        = "HealthyTaskCount"
     namespace          = "AWS/ECS"
     period             = "300"
     statistic          = "Average"
     threshold          = "1"
     
     dimensions = {
       ClusterName = aws_ecs_cluster.main.name
       ServiceName = "gen-ai-service"
     }
     
     alarm_description = "Service health check"
     alarm_actions     = [aws_sns_topic.alerts.arn]
     
     tags = {
       Environment = "production"
       Project     = "gen-ai-services"
     }
   }
   ```

2. Infrastructure Alerts
   - CPU > 80%
   - Memory > 85%
   - Disk Space < 20%
   - Error Rate > 5%

## Backup and Recovery

### Database Backups

1. Automated Backups
   - Daily backups
   - 7-day retention
   - Point-in-time recovery

2. Manual Backups
   - Before major changes
   - Before deployments
   - On demand

### Disaster Recovery

1. Multi-Region Setup
   - Primary region: us-east-1
   - Secondary region: us-west-2
   - Cross-region replication

2. Recovery Procedures
   - Database failover
   - Service failover
   - Data restoration

## Cost Optimization

### Resource Optimization

1. Auto Scaling
   ```hcl
   resource "aws_appautoscaling_target" "service" {
     max_capacity       = 4
     min_capacity       = 1
     resource_id        = "service/${aws_ecs_cluster.main.name}/${aws_ecs_service.main.name}"
     scalable_dimension = "ecs:service:DesiredCount"
     service_namespace  = "ecs"
   }
   ```

2. Reserved Instances
   - 1-year commitment
   - No upfront payment
   - Significant savings

### Cost Monitoring

1. Budget Alerts
   ```hcl
   resource "aws_budgets_budget" "monthly" {
     name              = "gen-ai-monthly"
     budget_type       = "COST"
     limit_amount      = "1000"
     limit_unit        = "USD"
     time_unit         = "MONTHLY"
     
     cost_filters = {
       TagKeyValue = ["Project$gen-ai-services"]
     }
     
     notification {
       comparison_operator = "GREATER_THAN"
       threshold          = 80
       threshold_type     = "PERCENTAGE"
       notification_type  = "ACTUAL"
     }
   }
   ```

2. Cost Reports
   - Monthly reports
   - Service breakdown
   - Resource utilization
   - Optimization recommendations

## Security

### Network Security

1. VPC Endpoints
   ```hcl
   resource "aws_vpc_endpoint" "s3" {
     vpc_id       = module.vpc.vpc_id
     service_name = "com.amazonaws.us-east-1.s3"
     
     tags = {
       Environment = "production"
       Project     = "gen-ai-services"
     }
   }
   ```

2. Security Groups
   - Minimal access
   - Service-specific rules
   - Regular review

### Data Security

1. Encryption
   - At rest: AES-256
   - In transit: TLS 1.2
   - Key rotation

2. Access Control
   - IAM roles
   - Resource policies
   - Regular audit

## Compliance

### Audit Trail

1. CloudTrail
   ```hcl
   resource "aws_cloudtrail" "main" {
     name                          = "gen-ai-trail"
     s3_bucket_name               = aws_s3_bucket.cloudtrail.id
     include_global_service_events = true
     is_multi_region_trail        = true
     
     event_selector {
       read_write_type           = "All"
       include_management_events = true
     }
     
     tags = {
       Environment = "production"
       Project     = "gen-ai-services"
     }
   }
   ```

2. Logging
   - Service logs
   - Access logs
   - Audit logs

### Compliance Controls

1. Data Protection
   - PII handling
   - Data retention
   - Data deletion

2. Access Management
   - Role-based access
   - Regular review
   - Access logging 