# Changelog

All notable changes to the Strands RDS Discovery Tool will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.1.2] - 2025-10-05

### Fixed
- **Dual Logging Issue**: Removed persistent `rds_discovery.log` file creation
- **Clean Output**: Now only creates timestamped log files per assessment
- **No File Clutter**: Eliminated duplicate logging that created unnecessary files

## [2.1.1] - 2025-10-05

### Fixed
- **Single Log File**: Removed duplicate logging - now only creates timestamped log files per assessment
- **Clean Output**: No more persistent `rds_discovery.log` file cluttering the directory

## [2.1.0] - 2025-10-05

### Changed
- **Simplified Usage**: Removed action parameter - tool now works like original PowerShell script
- **Direct Assessment**: Always performs full assessment without needing to specify action='assess'
- **Streamlined Interface**: Function signature simplified to match PowerShell tool behavior

### Added
- **RUN_GUIDE.md**: Comprehensive step-by-step execution guide with virtual environment setup
- **Enhanced Documentation**: Updated all guides to reflect simplified usage

### Removed
- **Action Parameter**: No longer need to specify action='assess', 'template', 'explain', or 'recommend'
- **Complex Workflows**: Simplified to single assessment function like original PowerShell

## [2.0.2] - 2025-10-05

### Fixed
- **10% Tolerance Logic**: Fixed fallback function to properly detect within_tolerance matches
- **Match Type Accuracy**: Both AWS API and fallback paths now consistently apply 10% tolerance logic
- **Consistent Behavior**: Tolerance detection works regardless of AWS API credential availability

## [2.0.1] - 2025-10-05

### Changed
- **Output File Naming**: Standardized to `RDSdiscovery_[timestamp]` format for all assessments
- **Consistent Naming**: Same naming convention for single and multiple server assessments
- **File Format**: `RDSdiscovery_1759703573.csv`, `RDSdiscovery_1759703573.json`, `RDSdiscovery_1759703573.log`

### Fixed
- **Naming Consistency**: Removed variable naming based on input parameters
- **User Experience**: Predictable output file names regardless of assessment scope

## [2.0.0] - 2025-10-05

### Added
- **Pricing Integration**: Real-time AWS RDS pricing with hourly and monthly cost estimates
- **Triple Output Format**: CSV + JSON + LOG files with matching timestamps
- **AWS Instance Scaling Explanations**: exact_match, scaled_up, closest_fit, fallback logic
- **Enhanced JSON Output**: Metadata, pricing summary, and performance metrics
- **Complete Logging**: Success/failure documentation in dedicated log files
- **Cost Estimation**: Fallback pricing when AWS API unavailable
- **Pricing Summary**: Total monthly costs for batch assessments
- **Regional Pricing Support**: Defaults to us-east-1 with regional awareness
- **Multi-Server Assessment**: Batch processing with progress tracking and performance metrics

### Enhanced
- **AWS Instance Recommendations**: Improved scaling logic with cost impact analysis
- **Feature Detection**: Confirmed SSIS, SSRS, Always On AG as RDS compatible
- **PowerShell Compatibility**: Enhanced CSV output matching original tool exactly
- **Error Handling**: Comprehensive error logging and recovery
- **Performance Monitoring**: Detailed timing and success rate tracking
- **Return Value**: Simplified JSON response with file locations and summary
- **Batch Processing**: Sequential server assessment with real-time progress updates

### Fixed
- **Test Runner**: Fixed JSON parsing issues in test display
- **Storage Calculation**: Improved PowerShell-compatible xp_fixeddrives logic
- **Feature Blocking**: Corrected RDS blocking logic to match PowerShell exactly
- **Instance Sizing**: Fixed AWS instance family selection for high memory scenarios
- **Output File Naming**: Consistent RDSdiscovery_[timestamp] format

### Technical
- **AWS Pricing API**: Integration with real-time pricing data
- **Fallback Pricing**: Built-in pricing estimates for offline operation
- **Instance Families**: Complete support for m6i, r6i, x2iedn families
- **Output Management**: Coordinated file generation with timestamps
- **Logging Framework**: Enhanced logging with structured output
- **Performance Optimization**: Sub-2 second assessment times per server

### Validation
- **Multi-Server Testing**: Successfully tested with 16 server batch assessment
- **Performance Metrics**: 30.39 seconds for 16 servers (1.9s average per server)
- **Success Rate**: 100% success rate for multi-server assessments
- **Cost Calculation**: Accurate pricing aggregation ($8,994.88 for 16 servers)
- **Output Generation**: All 3 files generated correctly for batch assessments

## [1.0.0] - 2025-09-30

### Added
- **Initial Release**: SQL Server to AWS RDS migration assessment tool
- **PowerShell Compatibility**: CSV output matching original PowerShell RDS Discovery tool
- **Real SQL Server Data**: Live data collection from SQL Server instances
- **Feature Detection**: 25+ SQL Server feature compatibility checks
- **AWS Instance Recommendations**: Basic CPU-based instance sizing
- **Batch Processing**: Multiple server assessment capability
- **Authentication Support**: Windows and SQL Server authentication
- **Error Handling**: Production-ready error handling and logging

### Features
- **RDS Compatibility Assessment**: Identifies blocking features
- **Storage Calculation**: PowerShell-compatible storage logic using xp_fixeddrives
- **SSIS/SSRS Detection**: Proper filtering of system packages
- **Always On Support**: Correct handling of Always On AG and FCI
- **Service Broker**: Proper detection and compatibility assessment
- **CSV Output**: 41-column PowerShell-compatible format
- **JSON Output**: Detailed assessment data for programmatic use

### Technical Implementation
- **SQL Server Connectivity**: pyodbc-based connection management
- **Query Framework**: Structured SQL queries for feature detection
- **AWS Integration**: Basic boto3 integration for instance recommendations
- **Configuration Management**: Flexible timeout and authentication settings
- **Performance Optimization**: Efficient batch processing and connection pooling

## [Unreleased]

### Planned Features
- **Reserved Instance Pricing**: Integration with RI pricing models
- **Multi-Region Cost Comparison**: Regional pricing analysis
- **Storage Cost Estimation**: Complete TCO calculations
- **Cost Optimization Recommendations**: Automated right-sizing suggestions
- **Enhanced Reporting**: Executive summary and detailed technical reports
- **API Enhancements**: RESTful API for integration with other tools

---

## Version History Summary

- **v2.0.1**: Output file naming standardization
- **v2.0.0**: Major release with pricing integration and enhanced output formats
- **v1.0.0**: Initial release with core assessment functionality

## Migration Guide

### Upgrading from v2.0.0 to v2.0.1

**Changes:**
- Output file naming changed to consistent `RDSdiscovery_[timestamp]` format
- No functional changes to assessment logic or pricing

**Migration Steps:**
1. Update any scripts that depend on specific output file names
2. Adjust file parsing logic to use new naming convention
3. No code changes required for function calls

### Upgrading from v1.0.0 to v2.0.0

**Breaking Changes:**
- Return value format changed to include file locations
- Output now generates 3 files instead of 2
- Function signature unchanged (backward compatible)

**New Features:**
- Pricing data now included in all assessments
- Log files automatically generated
- Enhanced JSON metadata and summaries

**Migration Steps:**
1. Update code to handle new return value format
2. Adjust file handling for 3-file output structure
3. Update any parsing logic for enhanced JSON format
4. Review pricing data integration points

**Backward Compatibility:**
- All function parameters remain the same
- CSV output format unchanged
- Core assessment logic preserved
- Authentication methods unchanged
