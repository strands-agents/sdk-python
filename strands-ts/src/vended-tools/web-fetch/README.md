# Web Fetch Tool

Fetches an HTTP(S) URL and returns its readable content as markdown suitable for a model to read. Distinct from the http-request tool, which returns raw response bodies for API calls.

The tool is intentionally strict at the URL boundary because the model chooses the target. Only http and https schemes are accepted. Every host is DNS-resolved and every returned address is required to be publicly routable, so private, loopback, link-local, cloud metadata, CGNAT, multicast, and reserved ranges are refused; IPv4-mapped IPv6 addresses are unwrapped first. The tool then connects to a specific already-validated IP address, so a DNS rebinder cannot substitute a private address between validation and connect. HTTPS certificate verification still uses the public hostname via SNI. Every hop of a redirect chain goes through the same scheme and address checks, the response body is capped at five mebibytes, and the extracted markdown drops scripts, styles, iframes, and other active elements as well as data URI images and javascript URLs.

## Usage

```typescript
import { Agent } from '@strands-agents/sdk'
import { webFetch } from '@strands-agents/sdk/vended-tools/web-fetch'

const agent = new Agent({ tools: [webFetch] })
await agent.invoke('Summarize https://example.com/blog/post')
```

Direct invocation:

```typescript
const result = await webFetch.invoke({ url: 'https://example.com/' })
console.log(result.title)
console.log(result.markdown)
```

## API

### Input

| Property  | Type     | Required | Default | Description                                    |
| --------- | -------- | -------- | ------- | ---------------------------------------------- |
| `url`     | `string` | Yes      |         | URL to fetch. Must be `http://` or `https://`. |
| `timeout` | `number` | No       | 30      | Total request timeout in seconds.              |

### Output

| Property      | Type     | Description                                        |
| ------------- | -------- | -------------------------------------------------- |
| `url`         | `string` | Final URL after any redirects.                     |
| `status`      | `number` | HTTP status code of the final response.            |
| `contentType` | `string` | `Content-Type` header of the final response.       |
| `title`       | `string` | Extracted `<title>`, or empty if none was found.   |
| `markdown`    | `string` | Cleaned markdown extracted from the response body. |

For non-HTML responses (JSON, plain text), `markdown` is the decoded body verbatim and `title` is empty.

## What it does not do

This is a one-shot GET: no caching, cookies, or auth handling. JavaScript is not executed, so dynamic pages that build their content from client-side scripts will return only the initial HTML shell; use a headless browser tool if you need the rendered page. There is no robots.txt handling here either. That policy belongs at a higher level, in agent hooks or the caller's own gating.
