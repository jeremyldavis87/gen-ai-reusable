# Deployment Guide

## Prerequisites

- AWS CLI configured with appropriate credentials
- Docker installed locally
- kubectl (if using EKS)
- Terraform (for infrastructure deployment)

## Infrastructure Deployment

1. Navigate to the infrastructure directory:
   ```bash
   cd deploy/infrastructure
   ```

2. Initialize Terraform:
   ```bash
   terraform init
   ```

3. Review the deployment plan:
   ```bash
   terraform plan
   ```

4. Apply the infrastructure:
   ```bash
   terraform apply
   ```

## Service Deployment

### Local Development

1. Build and run services locally:
   ```bash
   docker-compose up --build
   ```

2. Access services at their respective ports:
   - Format Service: http://localhost:8001
   - Classification Service: http://localhost:8002
   - etc.

### AWS Deployment

1. Configure AWS credentials:
   ```bash
   aws configure
   ```

2. Build and push Docker images:
   ```bash
   ./scripts/build_and_push.sh
   ```

3. Deploy services to ECS:
   ```bash
   ./scripts/deploy_services.sh
   ```

## Environment Configuration

1. Development:
   ```bash
   cp config/.env.sample config/.env.development
   # Edit config/.env.development with development values
   ```

2. Staging:
   ```bash
   cp config/.env.sample config/.env.staging
   # Edit config/.env.staging with staging values
   ```

3. Production:
   ```bash
   cp config/.env.sample config/.env.production
   # Edit config/.env.production with production values
   ```

## Monitoring Deployment

1. Access CloudWatch dashboard:
   ```bash
   aws cloudwatch get-dashboard --dashboard-name gen-ai-services
   ```

2. Check service health:
   ```bash
   aws ecs describe-services --cluster gen-ai-cluster
   ```

## Rollback Procedures

1. Identify the previous deployment:
   ```bash
   aws ecs describe-services --cluster gen-ai-cluster
   ```

2. Roll back to previous version:
   ```bash
   aws ecs update-service --cluster gen-ai-cluster --service <service-name> --task-definition <previous-task-definition>
   ```

## Troubleshooting

### Common Issues

1. Service Health Checks Failing
   - Check CloudWatch logs
   - Verify environment variables
   - Check service dependencies

2. Deployment Failures
   - Check ECS task definition
   - Verify IAM roles
   - Check resource limits

3. Performance Issues
   - Monitor CloudWatch metrics
   - Check service scaling rules
   - Verify database connections

### Logs and Monitoring

1. View service logs:
   ```bash
   aws logs get-log-events --log-group-name /ecs/<service-name>
   ```

2. Check metrics:
   ```bash
   aws cloudwatch get-metric-statistics --namespace ECS --metric-name CPUUtilization
   ```

## Security Considerations

1. Secrets Management
   - Use AWS Secrets Manager for sensitive data
   - Rotate credentials regularly
   - Implement least privilege access

2. Network Security
   - Use private subnets
   - Implement security groups
   - Enable VPC flow logs

3. Compliance
   - Regular security audits
   - Access logging
   - Data encryption at rest 