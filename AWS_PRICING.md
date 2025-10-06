# AWS RDS Pricing Integration - Strands RDS Discovery Tool v2.0

## Overview

The Strands RDS Discovery Tool v2.0 includes comprehensive AWS RDS pricing integration that provides real-time cost estimates for recommended instances, helping organizations plan migration budgets and make informed decisions.

## Pricing Features

### Real-Time Cost Estimation
- **Hourly rates** for all recommended instances
- **Monthly estimates** based on 24/7 usage
- **Currency support** (USD)
- **Regional pricing** (defaults to us-east-1)
- **Pricing source transparency** (AWS API vs fallback)

### Instance Scaling with Cost Impact
- **exact_match**: Perfect specification match with precise pricing
- **scaled_up**: Shows cost impact of scaling to meet minimum requirements
- **closest_fit**: Best available match with cost comparison
- **fallback**: Estimated pricing when AWS API unavailable

## Pricing Data Sources

### 1. AWS Pricing API (Primary)
When AWS credentials are available, the tool uses real-time AWS Pricing API:

```python
# Automatic AWS API usage when credentials configured
result = strands_rds_discovery(
    action="assess",
    input_file="servers.csv",
    auth_type="sql",
    username="user",
    password="pass"
)
```

**Benefits:**
- Real-time current pricing
- Regional pricing accuracy
- Latest instance availability
- Reserved Instance pricing options

### 2. Fallback Pricing (Secondary)
When AWS API is unavailable, uses built-in pricing estimates:

**Advantages:**
- No AWS credentials required
- Consistent pricing baseline
- Offline capability
- Rapid assessment execution

## Supported Instance Pricing

### General Purpose (db.m6i)
| Instance Type | vCPU | Memory | Hourly Rate | Monthly Est. |
|---------------|------|--------|-------------|--------------|
| db.m6i.large | 2 | 8 GiB | $0.192 | $140.54 |
| db.m6i.xlarge | 4 | 16 GiB | $0.384 | $281.09 |
| db.m6i.2xlarge | 8 | 32 GiB | $0.768 | $562.18 |
| db.m6i.4xlarge | 16 | 64 GiB | $1.536 | $1,124.35 |
| db.m6i.8xlarge | 32 | 128 GiB | $3.072 | $2,248.70 |
| db.m6i.12xlarge | 48 | 192 GiB | $4.608 | $3,373.06 |
| db.m6i.16xlarge | 64 | 256 GiB | $6.144 | $4,497.41 |
| db.m6i.24xlarge | 96 | 384 GiB | $9.216 | $6,746.11 |

### Memory Optimized (db.r6i)
| Instance Type | vCPU | Memory | Hourly Rate | Monthly Est. |
|---------------|------|--------|-------------|--------------|
| db.r6i.large | 2 | 16 GiB | $0.252 | $184.31 |
| db.r6i.xlarge | 4 | 32 GiB | $0.504 | $368.62 |
| db.r6i.2xlarge | 8 | 64 GiB | $1.008 | $737.23 |
| db.r6i.4xlarge | 16 | 128 GiB | $2.016 | $1,474.46 |
| db.r6i.8xlarge | 32 | 256 GiB | $4.032 | $2,948.93 |
| db.r6i.16xlarge | 64 | 512 GiB | $8.064 | $5,897.86 |

### High Memory (db.x2iedn)
| Instance Type | vCPU | Memory | Hourly Rate | Monthly Est. |
|---------------|------|--------|-------------|--------------|
| db.x2iedn.large | 2 | 16 GiB | $0.668 | $488.79 |
| db.x2iedn.xlarge | 4 | 32 GiB | $1.336 | $977.58 |
| db.x2iedn.2xlarge | 8 | 64 GiB | $2.672 | $1,955.17 |
| db.x2iedn.4xlarge | 16 | 128 GiB | $5.344 | $3,910.34 |
| db.x2iedn.8xlarge | 32 | 256 GiB | $10.688 | $7,820.67 |
| db.x2iedn.16xlarge | 64 | 512 GiB | $21.376 | $15,641.34 |
| db.x2iedn.24xlarge | 96 | 768 GiB | $32.064 | $23,462.02 |

## Pricing Output Examples

### Individual Server Pricing
```json
{
  "server": "sql-prod-01",
  "aws_recommendation": {
    "instance_type": "db.m6i.4xlarge",
    "match_type": "scaled_up",
    "explanation": "Scaled up from 12 CPU/24GB to meet minimum requirements",
    "pricing": {
      "hourly_rate": 1.536,
      "monthly_estimate": 1124.35,
      "currency": "USD",
      "unit": "Hrs",
      "note": "Estimated pricing (fallback)"
    }
  }
}
```

### Batch Assessment Pricing Summary
```json
{
  "pricing_summary": {
    "total_monthly_cost": 4562.18,
    "currency": "USD",
    "note": "Costs are estimates and may vary by region and usage",
    "breakdown": {
      "rds_compatible_servers": 3,
      "total_servers_assessed": 5,
      "average_monthly_cost_per_server": 1520.73
    }
  }
}
```

## Cost Optimization Recommendations

### Right-Sizing Analysis
The tool provides intelligent sizing recommendations:

**Over-Provisioned Example:**
```
Current Server: 32 CPU, 64GB RAM
Recommendation: db.m6i.8xlarge (32 CPU, 128GB)
Monthly Cost: $2,248.70
Note: Memory upgrade included for optimal performance
```

**Under-Provisioned Example:**
```
Current Server: 4 CPU, 32GB RAM  
Recommendation: db.r6i.2xlarge (8 CPU, 64GB)
Monthly Cost: $737.23
Note: Scaled up CPU to meet minimum RDS requirements
```

### Cost Comparison Scenarios

#### Scenario 1: Small Development Environment
- **Servers**: 3 small SQL Servers (2-4 CPU each)
- **Recommendations**: db.m6i.large to db.m6i.xlarge
- **Total Monthly Cost**: $421.63 - $843.27
- **Migration Benefit**: Reduced infrastructure overhead

#### Scenario 2: Medium Production Environment  
- **Servers**: 5 medium SQL Servers (8-16 CPU each)
- **Recommendations**: db.m6i.2xlarge to db.m6i.4xlarge
- **Total Monthly Cost**: $2,810.90 - $5,621.75
- **Migration Benefit**: High availability, automated backups

#### Scenario 3: Large Enterprise Environment
- **Servers**: 10 large SQL Servers (16-64 CPU each)
- **Recommendations**: db.m6i.4xlarge to db.m6i.16xlarge
- **Total Monthly Cost**: $11,243.50 - $44,974.10
- **Migration Benefit**: Managed service, reduced operational overhead

## Regional Pricing Considerations

### Pricing Variations by Region
AWS RDS pricing varies by region. The tool defaults to **us-east-1** pricing:

**Example Regional Differences (db.m6i.2xlarge):**
- **us-east-1**: $0.768/hour
- **us-west-2**: $0.768/hour  
- **eu-west-1**: $0.845/hour
- **ap-southeast-1**: $0.922/hour

### Multi-Region Cost Planning
For multi-region deployments, consider:
- **Primary region**: Production workloads
- **Secondary region**: Disaster recovery
- **Data transfer costs**: Cross-region replication
- **Backup storage**: Regional backup pricing

## Cost Optimization Strategies

### 1. Reserved Instances
- **1-Year Term**: 30-40% savings over On-Demand
- **3-Year Term**: 50-60% savings over On-Demand
- **Payment Options**: All Upfront, Partial Upfront, No Upfront

### 2. Instance Family Selection
- **General Purpose (m6i)**: Balanced CPU/memory ratio
- **Memory Optimized (r6i)**: High memory workloads
- **High Memory (x2iedn)**: Extreme memory requirements

### 3. Storage Optimization
- **gp3 Storage**: Cost-effective general purpose
- **io2 Storage**: High IOPS requirements
- **Magnetic Storage**: Infrequent access patterns

## Pricing Accuracy and Disclaimers

### Accuracy Levels
1. **AWS API Pricing**: Real-time, highly accurate
2. **Fallback Pricing**: Estimated, updated periodically
3. **Regional Variations**: May differ from us-east-1 baseline

### Cost Factors Not Included
- **Storage costs**: Database storage pricing
- **Backup costs**: Automated backup storage
- **Data transfer**: Cross-AZ and internet transfer
- **Monitoring**: CloudWatch and enhanced monitoring
- **Licensing**: SQL Server license costs (included in RDS)

### Important Notes
- Pricing estimates are for **On-Demand instances**
- **Reserved Instance** pricing can provide significant savings
- **Spot instances** not available for RDS
- Costs may vary based on **actual usage patterns**
- **Storage and backup** costs are additional

## Integration with Migration Planning

### Budget Planning
Use pricing data for:
- **Migration budget estimation**
- **ROI calculations**
- **Cost comparison** (on-premises vs RDS)
- **Resource planning** and capacity management

### Decision Making
Pricing information helps with:
- **Instance type selection**
- **Migration prioritization** (cost vs complexity)
- **Timeline planning** based on budget constraints
- **Stakeholder communication** with concrete cost data

### Reporting and Documentation
The tool provides pricing data in multiple formats:
- **CSV reports**: For spreadsheet analysis
- **JSON data**: For programmatic processing
- **Summary reports**: For executive presentations
- **Detailed logs**: For audit and compliance

## Future Enhancements

### Planned Pricing Features
- **Reserved Instance pricing** integration
- **Multi-region cost comparison**
- **Storage cost estimation**
- **Total Cost of Ownership (TCO)** calculations
- **Cost trend analysis** and forecasting

### API Enhancements
- **Real-time pricing updates**
- **Custom pricing scenarios**
- **Cost optimization recommendations**
- **Budget alerting and monitoring**
