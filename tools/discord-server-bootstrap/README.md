# Discord Server Bootstrap (SeekingBeta.AI)

Config-driven Discord bot setup to quickly create:
- Roles
- Categories/channels
- Permission overwrites
- Rules panel with **I Agree** button
- Welcome message flow
- Admin slash commands for re-bootstrap

## 1) Create the Discord bot (one-time)
1. Open Discord Developer Portal.
2. Create a new application -> Bot.
3. Enable intents:
   - `SERVER MEMBERS INTENT` (required for welcome events)
4. Copy bot token.
5. Invite bot to your server with scopes:
   - `bot`
   - `applications.commands`
6. Recommended bot permissions:
   - Manage Roles
   - Manage Channels
   - View Channels
   - Send Messages
   - Embed Links
   - Read Message History

## 2) Configure project
```bash
cd tools/discord-server-bootstrap
cp .env.example .env
```

Update `.env`:
- `DISCORD_BOT_TOKEN`
- `DISCORD_GUILD_ID`

Update `config/server-config.json`:
- Role names/colors
- Categories/channels
- Rules content
- Welcome message
- Discord invite URL

## 3) Install and run
```bash
npm install
npm run once
```

`once` mode logs in, registers commands, bootstraps, and exits.

Run continuously for buttons/welcome flow:
```bash
npm run start
```

## 4) Slash commands (admin only)
- `/bootstrap` -> sync roles/channels/rules panel
- `/repost-rules` -> update rules panel in `#rules`
- `/ping` -> health check

## Notes
- Setup is idempotent by role/channel name.
- If role/channel names are changed in config, rerun `/bootstrap`.
- Rules acceptance grants the role configured in `acceptRoleName` (default: `Member`).
- Keep the bot online (`npm run start`) for:
  - rules button interaction handling
  - welcome messages for new members
