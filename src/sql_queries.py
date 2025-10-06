"""
SQL Server Assessment Queries
Ported from PowerShell RDS Discovery Tool LimitationQueries.sql
"""

# Basic server information
SERVER_INFO_QUERY = """
SELECT 
    SERVERPROPERTY('Edition') AS Edition,
    SERVERPROPERTY('ProductVersion') AS ProductVersion,
    CAST(SERVERPROPERTY('IsClustered') AS INT) AS IsClustered
"""

# CPU and Memory information
CPU_MEMORY_QUERY = """
SELECT 
    cpu_count AS CPU,
    (SELECT CONVERT(int, value_in_use)/1024 
     FROM sys.configurations 
     WHERE name LIKE 'max server memory%') AS MaxMemory
FROM sys.dm_os_sys_info WITH (NOLOCK)
"""

# Database size information
DATABASE_SIZE_QUERY = """
SELECT 
    ISNULL(ROUND(SUM((CAST(size AS BIGINT) * 8))/1024.0/1024.0, 2), 0) AS TotalSizeGB
FROM sys.master_files 
WHERE database_id > 4
"""

# Comprehensive feature compatibility checks - ported from PowerShell
FEATURE_CHECKS = {
    # Linked Servers (non-SQL Server)
    "linked_servers": """
        SELECT CASE WHEN COUNT(*) = 0 THEN 'N' ELSE 'Y' END AS IsLinkedServer
        FROM sys.servers 
        WHERE is_linked = 1 AND product <> 'SQL Server' AND product <> 'oracle'
    """,
    
    # FileStream
    "filestream": """
        SELECT CASE WHEN value_in_use = 0 THEN 'N' ELSE 'Y' END AS IsFilestream
        FROM sys.configurations 
        WHERE name LIKE 'filestream%'
    """,
    
    # Resource Governor
    "resource_governor": """
        SELECT CASE WHEN classifier_function_id = 0 THEN 'N' ELSE 'Y' END AS IsResourceGov
        FROM sys.dm_resource_governor_configuration
    """,
    
    # Log Shipping
    "log_shipping": """
        SELECT CASE 
            WHEN EXISTS (SELECT 1 FROM msdb.dbo.log_shipping_primary_databases) THEN 'Y'
            ELSE 'N' 
        END AS IsLogShipping
    """,
    
    # Service Broker Endpoints
    "service_broker": """
        SELECT CASE WHEN COUNT(*) = 0 THEN 'N' ELSE 'Y' END AS IsServiceBroker
        FROM sys.service_broker_endpoints
    """,
    
    # Database Count > 100
    "database_count": """
        SELECT CASE WHEN COUNT(*) > 100 THEN 'Y' ELSE 'N' END AS IsDBCount
        FROM sys.databases 
        WHERE database_id > 4
    """,
    
    # Transaction Replication
    "transaction_replication": """
        SELECT CASE 
            WHEN EXISTS (
                SELECT 1 FROM sys.databases 
                WHERE database_id > 4 
                AND (is_published = 1 OR is_merge_published = 1 OR is_distributor = 1)
            ) THEN 'Y'
            ELSE 'N' 
        END AS IsTransReplication
    """,
    
    # Extended Procedures (non-standard)
    "extended_procedures": """
        SELECT CASE WHEN COUNT(*) = 0 THEN 'N' ELSE 'Y' END AS IsExtendedProc
        FROM master.sys.extended_procedures
    """,
    
    # TSQL Endpoints
    "tsql_endpoints": """
        SELECT CASE WHEN COUNT(*) = 0 THEN 'N' ELSE 'Y' END AS IsTSQLEndpoint
        FROM sys.routes 
        WHERE address != 'LOCAL'
    """,
    
    # PolyBase (SQL Server 2016+)
    "polybase": """
        SELECT CASE 
            WHEN SUBSTRING(CONVERT(CHAR(5), SERVERPROPERTY('ProductVersion')), 1, 2) < '13' THEN 'Not Supported'
            WHEN COUNT(*) = 0 THEN 'N' 
            ELSE 'Y' 
        END AS IsPolyBase
        FROM sys.external_data_sources
    """,
    
    # Buffer Pool Extension (SQL Server 2014+)
    "buffer_pool_extension": """
        SELECT CASE 
            WHEN SUBSTRING(CONVERT(CHAR(5), SERVERPROPERTY('ProductVersion')), 1, 2) < '12' THEN 'Not Supported'
            WHEN COUNT(*) = 0 THEN 'N' 
            ELSE 'Y' 
        END AS IsBufferPoolExt
        FROM sys.dm_os_buffer_pool_extension_configuration
        WHERE [state] != 0
    """,
    
    # File Tables (SQL Server 2012+)
    "file_tables": """
        SELECT CASE 
            WHEN SUBSTRING(CONVERT(CHAR(5), SERVERPROPERTY('ProductVersion')), 1, 2) = '10' THEN 'Not Supported'
            WHEN EXISTS (SELECT 1 FROM sys.tables WHERE is_filetable = 1) THEN 'Y'
            ELSE 'N' 
        END AS IsFileTable
    """,
    
    # Stretch Database
    "stretch_database": """
        SELECT CASE WHEN value = 0 THEN 'N' ELSE 'Y' END AS IsStretchDB
        FROM sys.configurations 
        WHERE name LIKE 'remote data archive'
    """,
    
    # Trustworthy Databases
    "trustworthy_databases": """
        SELECT CASE WHEN COUNT(*) = 0 THEN 'N' ELSE 'Y' END AS IsTrustworthy
        FROM sys.databases 
        WHERE database_id > 4 AND is_trustworthy_on > 0
    """,
    
    # Server Triggers
    "server_triggers": """
        SELECT CASE WHEN COUNT(*) = 0 THEN 'N' ELSE 'Y' END AS IsServerTrigger
        FROM sys.server_triggers
    """,
    
    # R and Machine Learning Services
    "machine_learning": """
        SELECT CASE WHEN value = 0 THEN 'N' ELSE 'Y' END AS IsMachineLearning
        FROM sys.configurations 
        WHERE name LIKE 'external scripts enabled'
    """,
    
    # Data Quality Services
    "data_quality_services": """
        SELECT CASE WHEN COUNT(*) = 0 THEN 'N' ELSE 'Y' END AS IsDQS
        FROM sys.databases 
        WHERE name LIKE 'DQS%'
    """,
    
    # Policy Based Management
    "policy_based_management": """
        SELECT CASE WHEN COUNT(*) = 0 THEN 'N' ELSE 'Y' END AS IsPolicyBased
        FROM msdb.dbo.syspolicy_policy_execution_history_details
    """,
    
    # CLR Enabled (version dependent)
    "clr_enabled": """
        SELECT CASE 
            WHEN value_in_use = 1 AND SUBSTRING(CONVERT(CHAR(5), SERVERPROPERTY('ProductVersion')), 1, 2) <= '13' THEN 'N'
            WHEN value_in_use = 1 AND SUBSTRING(CONVERT(CHAR(5), SERVERPROPERTY('ProductVersion')), 1, 2) > '13' THEN 'Y'
            ELSE 'N'
        END AS IsCLREnabled
        FROM sys.configurations 
        WHERE name LIKE 'clr enabled%'
    """,
    
    # Always On Availability Groups
    "always_on_ag": """
        SELECT CASE 
            WHEN SERVERPROPERTY('IsHadrEnabled') = 1 THEN 'Y' 
            ELSE 'N' 
        END AS IsAlwaysOnAG
    """,
    
    # Always On Failover Cluster Instance
    "always_on_fci": """
        SELECT CASE 
            WHEN SERVERPROPERTY('IsClustered') = 1 THEN 'Y' 
            ELSE 'N' 
        END AS IsAlwaysOnFCI
    """,
    
    # Server Role (Primary/Secondary/Standalone)
    "server_role": """
        SELECT CASE 
            WHEN SERVERPROPERTY('IsHadrEnabled') = 0 THEN 'Standalone'
            WHEN EXISTS (SELECT 1 FROM sys.dm_hadr_availability_replica_states 
                        WHERE is_local = 1 AND role_desc = 'PRIMARY') THEN 'Primary'
            WHEN EXISTS (SELECT 1 FROM sys.dm_hadr_availability_replica_states 
                        WHERE is_local = 1 AND role_desc = 'SECONDARY') THEN 'Secondary'
            ELSE 'Standalone'
        END AS ServerRole
    """,
    
    # Read Only Replica
    "read_only_replica": """
        SELECT CASE 
            WHEN SERVERPROPERTY('IsHadrEnabled') = 0 THEN 'N'
            WHEN EXISTS (
                SELECT 1 FROM sys.availability_replicas ar
                INNER JOIN sys.dm_hadr_availability_replica_states ars 
                ON ar.replica_id = ars.replica_id
                WHERE ars.is_local = 1 
                AND ar.secondary_role_allow_connections_desc IN ('READ_ONLY', 'ALL')
                AND ars.role_desc = 'SECONDARY'
            ) THEN 'Y'
            ELSE 'N'
        END AS IsReadReplica
    """,
    
    # Enterprise Features Detection
    "enterprise_features": """
        SELECT CASE 
            WHEN EXISTS (SELECT 1 FROM sys.dm_db_persisted_sku_features) THEN 'Y'
            ELSE 'N'
        END AS HasEnterpriseFeatures
    """,
    
    # Online Index Operations (Enterprise feature)
    "online_indexes": """
        SELECT CASE 
            WHEN CAST(SERVERPROPERTY('Edition') AS VARCHAR(100)) LIKE '%Enterprise%' 
            AND EXISTS (SELECT 1 FROM sys.dm_db_persisted_sku_features 
                       WHERE feature_name LIKE '%OnlineIndexOperation%') THEN 'Y'
            ELSE 'N'
        END AS IsOnlineIndexes
    """,
    
    # SSIS Detection - Check if SSIS is actually enabled (exclude all default system packages)
    "ssis": """
        SELECT CASE 
            WHEN EXISTS (SELECT 1 FROM sys.databases WHERE name = 'SSISDB')
            OR EXISTS (SELECT 1 FROM msdb.dbo.sysssispackages 
                      WHERE name NOT LIKE 'Maintenance%' 
                      AND name NOT LIKE 'Data Collector%'
                      AND name NOT LIKE 'PerfCounters%'
                      AND name NOT LIKE 'QueryActivity%'
                      AND name NOT LIKE 'SqlTrace%'
                      AND name NOT LIKE 'ServerActivity%'
                      AND name NOT LIKE 'DiskUsage%'
                      AND name NOT LIKE 'TSQLQuery%')
            THEN 'Y' 
            ELSE 'N' 
        END AS IsSSIS
    """,
    
    # SSRS Detection  
    "ssrs": """
        SELECT CASE 
            WHEN EXISTS (SELECT 1 FROM sys.databases WHERE name LIKE 'ReportServer%')
            OR EXISTS (SELECT 1 FROM sys.databases WHERE name = 'ReportServerTempDB')
            THEN 'Y' 
            ELSE 'N' 
        END AS IsSSRS
    """
}

# Additional queries for enhanced assessment
PERFORMANCE_QUERIES = {
    # ElastiCache recommendation based on read/write patterns
    "elasticache_recommendation": """
        WITH Read_WriteIO AS (
            SELECT 
                qs.total_logical_reads,
                qs.total_logical_writes,
                (qs.total_logical_reads * 8 / 1024.0) AS [Total Logical Reads (MB)]
            FROM sys.dm_exec_query_stats AS qs
        ),
        ReadOverWrite AS (
            SELECT TOP 10
                total_logical_reads,
                total_logical_writes,
                ([Total Logical Reads (MB)] * 100) / 
                    (SELECT SUM([Total Logical Reads (MB)]) FROM Read_WriteIO) AS overallreadweight,
                (total_logical_reads * 100) / 
                    NULLIF(total_logical_reads + total_logical_writes, 0) AS readoverwriteweight
            FROM Read_WriteIO 
            ORDER BY overallreadweight DESC
        )
        SELECT CASE 
            WHEN AVG(readoverwriteweight) > 90 THEN 'Y'
            ELSE 'N' 
        END AS RecommendElastiCache
        FROM ReadOverWrite
    """,
    
    # Source detection (RDS, GCP, EC2/OnPrem)
    "source_detection": """
        SELECT CASE 
            WHEN EXISTS (SELECT 1 FROM sys.databases WHERE name = 'rdsadmin') THEN 'RDS'
            WHEN EXISTS (SELECT 1 FROM sys.databases WHERE name LIKE 'gcp%') THEN 'GCP'
            ELSE 'EC2/OnPrem'
        END AS Source
    """
}

# Queries that require special handling or multiple databases
COMPLEX_QUERIES = {
    # This requires iteration through all databases
    "subscription_replication": """
        -- Check for subscription replication across all databases
        -- Note: This needs to be executed per database in Python code
        SELECT CASE 
            WHEN OBJECT_ID('dbo.syssubscriptions', 'U') IS NOT NULL THEN 'Y'
            ELSE 'N'
        END AS HasSubscriptions
    """,
    
    # Enterprise features across all databases
    "enterprise_features_detailed": """
        -- Check for enterprise features across all databases
        -- Note: This needs to be executed per database in Python code
        SELECT 
            DB_NAME() AS DatabaseName,
            feature_name,
            feature_id
        FROM sys.dm_db_persisted_sku_features
    """
}
