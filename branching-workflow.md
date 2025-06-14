graph TD
    subgraph External Repositories
        U[Strands Agents<br/>upstream/main] -->|Daily Sync| B
        D[Dify] -->|Frontend Integration| F1
        S[Smolagents] -->|Feature Integration| F2
    end

    subgraph StrandForge Organization
        B[base-main] -->|Merge Features| D
        D[develop] --> F1[feature/visual-builder]
        D --> F2[feature/k8s-operator]
        D --> F3[feature/mcp-enhancements]
        F1 -->|PR| D
        F2 -->|PR| D
        F3 -->|PR| D
        D --> R[release/v1.0.0]
        R -->|Tag| M[main]
        M --> H[hotfix/urgent-bug]
        H -->|PR| M
        H -->|Cherry-pick| D
    end

    style U stroke:#0366d6,stroke-width:2px
    style B stroke:#22863a,stroke-width:2px
    style D stroke:#6f42c1,stroke-width:2px
    style F1,F2,F3 stroke:#d73a49,stroke-width:1.5px
    style R stroke:#e36209,stroke-width:2px
    style M stroke:#005cc5,stroke-width:3px
