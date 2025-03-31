# Monitoring and Logging Guide

## Overview

The Gen AI Services platform uses AWS CloudWatch for comprehensive monitoring and logging. This guide covers the monitoring setup, metrics collection, and log management.

## CloudWatch Dashboard

### Key Metrics

1. Service Health
   - CPU Utilization
   - Memory Usage
   - Request Count
   - Error Rate
   - Response Time

2. Resource Usage
   - ECS Task Count
   - Container Health
   - Database Connections
   - Cache Hit Rate

3. Business Metrics
   - API Calls per Service
   - Processing Time
   - Success Rate
   - User Activity

## Logging Strategy

### Log Levels

- ERROR: Critical issues requiring immediate attention
- WARNING: Potential issues that need monitoring
- INFO: Normal operational information
- DEBUG: Detailed information for troubleshooting

### Log Structure

```json
{
  "timestamp": "2024-03-28T12:00:00Z",
  "level": "INFO",
  "service": "format_conversion_service",
  "trace_id": "abc123",
  "message": "Processing document",
  "metadata": {
    "document_id": "doc123",
    "format": "pdf",
    "size": 1024
  }
}
```

## Alerting

### Alert Rules

1. Service Health Alerts
   - CPU > 80% for 5 minutes
   - Memory > 85% for 5 minutes
   - Error rate > 5% for 5 minutes

2. Business Alerts
   - Response time > 2 seconds
   - Queue length > 1000
   - Failed jobs > 10

3. Infrastructure Alerts
   - Disk space < 20%
   - Network errors > 100/minute
   - Service unavailable

## Monitoring Tools

### CloudWatch Metrics

1. ECS Metrics
   ```bash
   aws cloudwatch get-metric-statistics \
     --namespace ECS \
     --metric-name CPUUtilization \
     --dimensions Name=ClusterName,Value=gen-ai-cluster
   ```

2. Application Metrics
   ```bash
   aws cloudwatch get-metric-statistics \
     --namespace GenAIServices \
     --metric-name RequestCount \
     --dimensions Name=Service,Value=format_service
   ```

### Log Analysis

1. View Logs
   ```bash
   aws logs get-log-events \
     --log-group-name /ecs/format-service \
     --log-stream-name ecs/task-definition/1
   ```

2. Search Logs
   ```bash
   aws logs filter-log-events \
     --log-group-name /ecs/format-service \
     --filter-pattern "ERROR"
   ```

## Performance Monitoring

### Key Performance Indicators (KPIs)

1. Response Time
   - P50: < 200ms
   - P95: < 500ms
   - P99: < 1000ms

2. Availability
   - Uptime: > 99.9%
   - Error Rate: < 1%

3. Resource Utilization
   - CPU: < 70%
   - Memory: < 80%
   - Disk: < 70%

## Cost Monitoring

### Cost Optimization

1. Resource Usage
   - Monitor ECS task scaling
   - Track S3 storage usage
   - Monitor database connections

2. Cost Alerts
   - Daily cost threshold
   - Unusual usage patterns
   - Resource optimization opportunities

## Troubleshooting

### Common Issues

1. High CPU Usage
   - Check for infinite loops
   - Monitor background tasks
   - Review scaling rules

2. Memory Leaks
   - Monitor container memory
   - Check for resource cleanup
   - Review garbage collection

3. Network Issues
   - Check security groups
   - Monitor connection limits
   - Review DNS resolution

## Best Practices

1. Logging
   - Use structured logging
   - Include correlation IDs
   - Implement log rotation
   - Set appropriate retention

2. Monitoring
   - Define clear SLOs
   - Set up automated alerts
   - Regular metric review
   - Document incident response

3. Performance
   - Regular load testing
   - Capacity planning
   - Performance optimization
   - Resource scaling 