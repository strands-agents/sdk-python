# AWS Instance Sizing Feature

## Overview
The RDS Discovery Tool now provides intelligent AWS RDS instance recommendations based on actual server CPU and memory specifications, integrated with 27+ SQL Server compatibility checks including SSIS and SSRS detection.

## How It Works

### 1. Data Collection
- Collects CPU core count from SQL Server
- Collects maximum memory (MB) from SQL Server
- Converts memory to GB for calculations

### 2. Sizing Logic

#### Instance Family Selection (Memory-to-CPU Ratio)
```
Ratio = Memory_GB / CPU_Cores

≤ 4:1   → db.m6i.* (General Purpose)
≤ 8:1   → db.m6i.* (Balanced)  
≤ 16:1  → db.r6i.* (Memory Optimized)
> 16:1  → db.x2iedn.* (High Memory)
```

#### Instance Size Selection (CPU-based)
```
≤ 2 cores   → large
≤ 4 cores   → xlarge
≤ 8 cores   → 2xlarge
≤ 16 cores  → 4xlarge
≤ 32 cores  → 8xlarge
≤ 48 cores  → 12xlarge
≤ 64 cores  → 16xlarge
≤ 96 cores  → 24xlarge
> 96 cores  → 32xlarge
```

#### High Memory Override
```
> 1000GB → db.x2iedn.*
> 2000GB → db.x2iedn.24xlarge
> 1500GB → db.x2iedn.16xlarge
```

### 3. Fallback Strategy

#### Primary: AWS Pricing API
- Queries live AWS pricing data
- Matches SQL Server edition (SE/EE)
- Finds exact or best-fit instances

#### Fallback: Hardcoded Logic
- Used when AWS API fails
- Based on proven sizing patterns
- Ensures recommendations always available

### 4. Match Types
- **exact_match**: Perfect CPU/memory match
- **within_tolerance**: Within 10% of available instance (both CPU and memory)
- **scaled_up**: Next size up to meet requirements
- **closest_fit**: Closest available match
- **fallback**: Hardcoded recommendation when no tolerance match found

## Example Output

```
💡 Instance Size: db.m6i.2xlarge
📝 Sizing Note: Recommended based on general sizing guidelines
```

## Implementation Details

### Functions Added
- `get_aws_instance_recommendation()` - Main sizing function
- `get_rds_instances_from_api()` - AWS API integration
- `find_best_instance()` - Matching logic
- `get_fallback_instance_recommendation()` - Hardcoded fallback

### Dependencies
- `boto3` - AWS SDK for Python
- AWS credentials (optional - falls back if unavailable)

### Error Handling
- AWS API failures → Fallback to hardcoded logic
- Missing credentials → Fallback to hardcoded logic
- Invalid data → Default to safe recommendations

## Benefits

### Before
```
💡 Instance Size: Consider db.r5.xlarge or db.r5.large based on your workload
```

### After
```
💡 Instance Size: db.m6i.2xlarge
📝 Sizing Note: Scaled up to meet your requirements
```

### Improvements
- **Specific recommendations** instead of generic suggestions
- **Based on actual server specs** not guesswork
- **Modern instance types** (m6i, r6i, x2iedn)
- **Intelligent family selection** based on memory patterns
- **Reliable fallback** when APIs unavailable

## Configuration

### AWS Credentials (Optional)
```bash
# For live API data
aws configure
# or
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
```

### No Configuration Required
- Tool works without AWS credentials
- Falls back to proven sizing logic
- Always provides recommendations
