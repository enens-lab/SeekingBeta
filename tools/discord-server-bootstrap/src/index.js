import fs from "node:fs";
import path from "node:path";
import dotenv from "dotenv";
import {
  ActionRowBuilder,
  ButtonBuilder,
  ButtonStyle,
  ChannelType,
  Client,
  EmbedBuilder,
  GatewayIntentBits,
  PermissionFlagsBits,
  SlashCommandBuilder,
} from "discord.js";

dotenv.config();

const BOT_TOKEN = process.env.DISCORD_BOT_TOKEN?.trim();
const GUILD_ID = process.env.DISCORD_GUILD_ID?.trim();
const AUTO_BOOTSTRAP = (process.env.AUTO_BOOTSTRAP || "true").toLowerCase() === "true";
const RULES_BUTTON_CUSTOM_ID = process.env.RULES_BUTTON_CUSTOM_ID || "rules_accept";
const CONFIG_PATH = path.resolve(
  process.cwd(),
  process.env.SERVER_CONFIG_PATH || "./config/server-config.json"
);
const ONCE_MODE = process.argv.includes("--once");

if (!BOT_TOKEN || !GUILD_ID) {
  console.error("Missing DISCORD_BOT_TOKEN or DISCORD_GUILD_ID in environment.");
  process.exit(1);
}

if (!fs.existsSync(CONFIG_PATH)) {
  console.error(`Server config file not found: ${CONFIG_PATH}`);
  process.exit(1);
}

/** @type {import("../config/server-config.json")} */
const serverConfig = JSON.parse(fs.readFileSync(CONFIG_PATH, "utf-8"));

const client = new Client({
  intents: [GatewayIntentBits.Guilds, GatewayIntentBits.GuildMembers],
});

function log(msg) {
  console.log(`[discord-bootstrap] ${msg}`);
}

function normalize(value) {
  return String(value || "").trim().toLowerCase();
}

function findRoleByName(guild, roleName) {
  const target = normalize(roleName);
  return guild.roles.cache.find((r) => normalize(r.name) === target) || null;
}

function findCategoryByName(guild, categoryName) {
  const target = normalize(categoryName);
  return (
    guild.channels.cache.find(
      (c) => c.type === ChannelType.GuildCategory && normalize(c.name) === target
    ) || null
  );
}

function findChannelByName(guild, channelName, type) {
  const target = normalize(channelName);
  return (
    guild.channels.cache.find(
      (c) => c.type === type && normalize(c.name) === target
    ) || null
  );
}

function resolvePermissionName(name) {
  const normalized = String(name || "")
    .trim()
    .replace(/[\s-]/g, "")
    .toLowerCase();
  return Object.keys(PermissionFlagsBits).find((key) => {
    const keyNorm = key.replace(/_/g, "").toLowerCase();
    return keyNorm === normalized;
  });
}

function resolvePermissionFlags(items) {
  const out = [];
  for (const raw of items || []) {
    const key = resolvePermissionName(raw);
    if (!key) {
      log(`Warning: unknown permission "${raw}" ignored`);
      continue;
    }
    out.push(PermissionFlagsBits[key]);
  }
  return out;
}

function buildPermissionOverwrites(guild, permissionConfig) {
  if (!permissionConfig) return undefined;
  const overwrites = [];
  const everyoneConfig = permissionConfig.everyone;
  if (everyoneConfig) {
    overwrites.push({
      id: guild.roles.everyone.id,
      allow: resolvePermissionFlags(everyoneConfig.allow),
      deny: resolvePermissionFlags(everyoneConfig.deny),
    });
  }

  for (const [roleName, rolePerms] of Object.entries(permissionConfig.roles || {})) {
    const role = findRoleByName(guild, roleName);
    if (!role) {
      log(`Warning: role "${roleName}" not found for permissions`);
      continue;
    }
    overwrites.push({
      id: role.id,
      allow: resolvePermissionFlags(rolePerms.allow),
      deny: resolvePermissionFlags(rolePerms.deny),
    });
  }
  return overwrites;
}

function channelTypeFromString(type) {
  if (normalize(type) === "voice") return ChannelType.GuildVoice;
  return ChannelType.GuildText;
}

async function ensureRole(guild, roleConfig) {
  const existing = findRoleByName(guild, roleConfig.name);
  const roleData = {
    name: roleConfig.name,
    color: roleConfig.color || null,
    hoist: Boolean(roleConfig.hoist),
    mentionable: Boolean(roleConfig.mentionable),
  };

  if (!existing) {
    const created = await guild.roles.create({
      ...roleData,
      reason: "SeekingBeta.AI server bootstrap",
    });
    log(`Created role: ${created.name}`);
    return created;
  }

  await existing.edit(roleData, "SeekingBeta.AI server bootstrap sync");
  log(`Synced role: ${existing.name}`);
  return existing;
}

async function ensureCategory(guild, categoryConfig) {
  let category = findCategoryByName(guild, categoryConfig.name);
  if (!category) {
    category = await guild.channels.create({
      name: categoryConfig.name,
      type: ChannelType.GuildCategory,
      reason: "SeekingBeta.AI server bootstrap",
    });
    log(`Created category: ${category.name}`);
  }
  return category;
}

async function ensureChannel(guild, category, channelConfig) {
  const channelType = channelTypeFromString(channelConfig.type);
  let channel = findChannelByName(guild, channelConfig.name, channelType);

  const channelData = {
    name: channelConfig.name,
    type: channelType,
    parent: category.id,
    topic: channelType === ChannelType.GuildText ? channelConfig.topic || null : undefined,
    permissionOverwrites: buildPermissionOverwrites(guild, channelConfig.permissions),
  };

  if (!channel) {
    channel = await guild.channels.create({
      ...channelData,
      reason: "SeekingBeta.AI server bootstrap",
    });
    log(`Created channel: #${channel.name}`);
  } else {
    await channel.edit(channelData, "SeekingBeta.AI server bootstrap sync");
    log(`Synced channel: #${channel.name}`);
  }

  return channel;
}

async function upsertRulesPanel(guild, config) {
  const rulesChannel = findChannelByName(guild, "rules", ChannelType.GuildText);
  if (!rulesChannel) {
    log('Warning: #rules channel not found; skipped rules panel');
    return;
  }

  const title = config.rules?.title || `${config.brandName || "Community"} Rules`;
  const description = config.rules?.description || "Please read and accept the rules.";
  const items = config.rules?.items || [];
  const footer = config.rules?.footer || "Click I Agree to continue.";

  const rulesLines = items.map((item, idx) => `${idx + 1}. ${item}`).join("\n");
  const inviteUrl = config.discordInviteUrl || "https://discord.gg/gydS5yb3";

  const embed = new EmbedBuilder()
    .setTitle(title)
    .setDescription(`${description}\n\n${rulesLines || ""}`.trim())
    .setColor("#2DD4BF")
    .addFields({ name: "Discord", value: `[Invite Link](${inviteUrl})` })
    .setFooter({ text: footer })
    .setTimestamp(new Date());

  const row = new ActionRowBuilder().addComponents(
    new ButtonBuilder()
      .setCustomId(RULES_BUTTON_CUSTOM_ID)
      .setStyle(ButtonStyle.Success)
      .setLabel("I Agree")
  );

  const messages = await rulesChannel.messages.fetch({ limit: 30 });
  const existing = messages.find((m) =>
    m.author.id === client.user.id &&
    m.components.some((componentRow) =>
      componentRow.components.some((component) => component.customId === RULES_BUTTON_CUSTOM_ID)
    )
  );

  if (existing) {
    await existing.edit({ embeds: [embed], components: [row] });
    log("Updated rules panel");
  } else {
    await rulesChannel.send({ embeds: [embed], components: [row] });
    log("Posted rules panel");
  }
}

async function bootstrapGuild(guild, config) {
  await guild.roles.fetch();
  await guild.channels.fetch();

  for (const roleConfig of config.roles || []) {
    await ensureRole(guild, roleConfig);
  }

  for (const categoryConfig of config.categories || []) {
    const category = await ensureCategory(guild, categoryConfig);
    for (const channelConfig of categoryConfig.channels || []) {
      await ensureChannel(guild, category, channelConfig);
    }
  }

  await guild.channels.fetch();
  await upsertRulesPanel(guild, config);
}

async function registerGuildCommands(guild) {
  const commands = [
    new SlashCommandBuilder()
      .setName("bootstrap")
      .setDescription("Sync roles/channels/rules panel from server config"),
    new SlashCommandBuilder()
      .setName("repost-rules")
      .setDescription("Repost or update the rules panel in #rules"),
    new SlashCommandBuilder()
      .setName("ping")
      .setDescription("Health check"),
  ];

  await guild.commands.set(commands.map((cmd) => cmd.toJSON()));
  log("Registered guild slash commands");
}

client.once("ready", async () => {
  log(`Logged in as ${client.user.tag}`);
  const guild = await client.guilds.fetch(GUILD_ID);
  await registerGuildCommands(guild);

  if (AUTO_BOOTSTRAP || ONCE_MODE) {
    await bootstrapGuild(guild, serverConfig);
  }

  if (ONCE_MODE) {
    log("Once mode complete. Exiting.");
    setTimeout(() => {
      client.destroy();
      process.exit(0);
    }, 600);
  }
});

client.on("guildMemberAdd", async (member) => {
  const welcomeChannelName = serverConfig.welcome?.channel;
  const template = serverConfig.welcome?.message;
  if (!welcomeChannelName || !template) return;

  const channel = findChannelByName(member.guild, welcomeChannelName, ChannelType.GuildText);
  if (!channel) return;

  const message = template.replaceAll("{user}", `<@${member.id}>`);
  await channel.send({ content: message });
});

client.on("interactionCreate", async (interaction) => {
  if (interaction.isButton() && interaction.customId === RULES_BUTTON_CUSTOM_ID) {
    const roleName = serverConfig.acceptRoleName || "Member";
    const role = findRoleByName(interaction.guild, roleName);

    if (!role) {
      await interaction.reply({
        content: `Role "${roleName}" is not configured yet. Please contact a moderator.`,
        ephemeral: true,
      });
      return;
    }

    if (!interaction.member?.roles?.add) {
      await interaction.reply({
        content: "Could not assign role. Please contact a moderator.",
        ephemeral: true,
      });
      return;
    }

    await interaction.member.roles.add(role, "Accepted rules via rules panel");
    await interaction.reply({
      content: `Thanks, you now have **${role.name}** access.`,
      ephemeral: true,
    });
    return;
  }

  if (!interaction.isChatInputCommand()) return;
  if (!interaction.memberPermissions?.has(PermissionFlagsBits.Administrator)) {
    await interaction.reply({
      content: "Only server administrators can run this command.",
      ephemeral: true,
    });
    return;
  }

  if (interaction.commandName === "ping") {
    await interaction.reply({ content: "pong", ephemeral: true });
    return;
  }

  if (interaction.commandName === "bootstrap") {
    await interaction.deferReply({ ephemeral: true });
    await bootstrapGuild(interaction.guild, serverConfig);
    await interaction.editReply("Bootstrap complete: roles, channels, and rules panel synced.");
    return;
  }

  if (interaction.commandName === "repost-rules") {
    await interaction.deferReply({ ephemeral: true });
    await upsertRulesPanel(interaction.guild, serverConfig);
    await interaction.editReply("Rules panel synced.");
  }
});

client.login(BOT_TOKEN);
