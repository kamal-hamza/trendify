"""
Management command to clean up useless/generic topics.
Deactivates topics that are not products, technologies, or valuable keywords.
"""

from django.core.management.base import BaseCommand

from api.models import Topic


class Command(BaseCommand):
    help = "Clean up useless/generic topics, keeping only valuable ones"

    def add_arguments(self, parser):
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Show what would be removed without actually removing",
        )
        parser.add_argument(
            "--auto",
            action="store_true",
            help="Automatically confirm removal without prompting",
        )

    def handle(self, *args, **options):
        dry_run = options["dry_run"]
        auto_confirm = options["auto"]

        self.stdout.write("\n" + "=" * 70)
        self.stdout.write("TOPIC CLEANUP")
        self.stdout.write("=" * 70)

        # Define rules for filtering
        GENERIC_REMOVE = {
            "other",
            "news",
            "project",
            "showcase",
            "discussion",
            "question",
            "help",
            "tutorial",
            "meta",
            "announcement",
            "general",
            "misleading",
            "society",
            "business",
            "politics",
            "showoff saturday",
            "miscellaneous",
            "resources",
            "social media",
            "transportation",
            "energy",
            "biotechnology",
            "antigravity",
            "weather",
            "usecase",
            "new model",
            "research",
        }

        # Noise/garbage topics
        NOISE_REMOVE = {
            "hit",
            "code",
            "code since",
            "since december",
            "december",
            "clawdbot",
            "clawdbot-security",
            "moltbot",
            "moltbot-skills",
            "openclaw",
            "openclaw-plugin",
            "openclaw-security",
            "openclaw-setup",
            "openclaw-skills",
            "openclawd",
            "opencode",
            "onecontext",
        }

        # Generic terms that should be removed
        GENERIC_TERMS = {
            "webdev",
            "coding",
            "technology",
            "software",
            "artificial",
        }

        ALL_REMOVE = GENERIC_REMOVE | NOISE_REMOVE | GENERIC_TERMS

        # Get all active topics
        all_topics = Topic.objects.filter(is_active=True)
        total_count = all_topics.count()

        self.stdout.write(f"\nTotal active topics: {total_count}")

        # Identify topics to deactivate
        to_deactivate = []
        for topic in all_topics:
            name_lower = topic.name.lower().strip()
            if name_lower in ALL_REMOVE:
                to_deactivate.append(topic)

        self.stdout.write(f"Topics to deactivate: {len(to_deactivate)}\n")

        if not to_deactivate:
            self.stdout.write(self.style.SUCCESS("\n✓ No useless topics found!"))
            return

        # Show what will be removed
        self.stdout.write("Topics to remove:")
        for topic in sorted(to_deactivate, key=lambda t: t.name):
            self.stdout.write(f"  ✗ {topic.name}")

        # Show what will be kept
        to_keep = [t for t in all_topics if t not in to_deactivate]
        self.stdout.write(f"\nValuable topics to keep: {len(to_keep)}")
        self.stdout.write("Sample of topics to keep:")
        for topic in sorted(to_keep, key=lambda t: t.name)[:15]:
            self.stdout.write(f"  ✓ {topic.name}")
        if len(to_keep) > 15:
            self.stdout.write(f"  ... and {len(to_keep) - 15} more")

        # Confirm and execute
        if dry_run:
            self.stdout.write(
                self.style.WARNING(
                    f"\n[DRY RUN] Would deactivate {len(to_deactivate)} topics"
                )
            )
            return

        # Prompt for confirmation
        if not auto_confirm:
            self.stdout.write(
                self.style.WARNING(
                    f"\nThis will deactivate {len(to_deactivate)} topics."
                )
            )
            confirm = input("Continue? (yes/no): ")
            if confirm.lower() != "yes":
                self.stdout.write("Cancelled.")
                return

        # Deactivate topics
        for topic in to_deactivate:
            topic.is_active = False
            topic.save()

        remaining = Topic.objects.filter(is_active=True).count()

        self.stdout.write("\n" + "=" * 70)
        self.stdout.write(self.style.SUCCESS("CLEANUP COMPLETE"))
        self.stdout.write("=" * 70)
        self.stdout.write(f"\n✓ Deactivated: {len(to_deactivate)} topics")
        self.stdout.write(f"✓ Remaining active: {remaining} topics")
        self.stdout.write(
            f"✓ Removed {(len(to_deactivate)/total_count)*100:.1f}% of topics\n"
        )