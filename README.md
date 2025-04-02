# HexDocs MCP Server

A TypeScript server implementing the Model Context Protocol (MCP) that provides semantic search capabilities for Hex package documentation. It's designed to work seamlessly with embeddings generated by the [hexdocs_mcp](https://github.com/bradleygolden/hexdocs-mcp) Elixir package, making Hex documentation easily searchable in AI applications.

## Features

- MCP-compatible server for searching Hex package documentation
- Integrates with embeddings generated by the [hexdocs_mcp](https://github.com/bradleygolden/hexdocs-mcp) Elixir package
- Built on the Model Context Protocol standard for compatibility with AI tools
- Easy installation via `npx` for quick setup
- Simple configuration with customizable database paths

## Requirements

- Node.js 18 or later
- Access to a [hexdocs_mcp](https://github.com/bradleygolden/hexdocs-mcp) generated SQLite database (requires running `mix hex.docs.mcp fetch PACKAGE` on the desired packages first)

## Configuration

The server looks for the SQLite database created by the `hexdocs_mcp` package. By default, it uses `~/.hexdocs_mcp/hexdocs_mcp.db`, but you can specify a custom path:

```bash
# Example: Set custom database path
export HEXDOCS_MCP_PATH=/path/to/custom/directory
```

## Integration

The server can be integrated with various AI tools that support the Model Context Protocol. Here's an example for Cursor:

Add this to your `mcp.json`:

```json
{
  "mcpServers": {
    "hexdocs-mcp-server": {
      "command": "npx",
      "args": [
        "-y",
        "hexdocs-mcp-server"
      ]
    }
  }
}
```

## Pro Tip

When you're vibing with an agent and you find that you don't have the given documentation for a specific tool, you can have the AI run the `mix hex.docs.mcp fetch ...` command for you so you don't have to.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

This project is licensed under MIT - see the [LICENSE](LICENSE) file for details.