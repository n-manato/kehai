import { Box, Button, Text } from '@chakra-ui/react';
import { memo, useCallback, useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import {
  DrawerRoot,
  DrawerTrigger,
  DrawerContent,
  DrawerHeader,
  DrawerTitle,
  DrawerBody,
  DrawerFooter,
  DrawerActionTrigger,
  DrawerBackdrop,
  DrawerCloseTrigger,
} from '@/components/ui/drawer';
import { sidebarStyles } from './sidebar-styles';

// A motion group and the display names of its clips, read from the loaded
// Live2D model's settings at drawer-open time.
interface MotionGroup {
  name: string;
  clips: { index: number; label: string }[];
}

interface MotionDrawerProps {
  children: React.ReactNode;
}

/** Turn "motions/AC_rem_mizugi01_act_egao.motion3.json" into "act_egao". */
const cleanClipLabel = (fileName: string): string => fileName
  .split('/')
  .pop()!
  .replace(/\.motion3\.json$/i, '')
  .replace(/^AC_[A-Za-z0-9]+_[A-Za-z0-9]+_/i, '');

/** Read every motion group + clip from the currently loaded model. */
const readMotionGroups = (): MotionGroup[] => {
  const model = (window as any).getLive2DManager?.()?.getModel?.(0);
  const setting = model?._modelSetting;
  if (!setting) return [];

  const groups: MotionGroup[] = [];
  const groupCount: number = setting.getMotionGroupCount();
  for (let i = 0; i < groupCount; i += 1) {
    const name: string = setting.getMotionGroupName(i);
    const clipCount: number = setting.getMotionCount(name);
    const clips = [] as MotionGroup['clips'];
    for (let j = 0; j < clipCount; j += 1) {
      clips.push({ index: j, label: cleanClipLabel(setting.getMotionFileName(name, j)) });
    }
    groups.push({ name, clips });
  }
  return groups;
};

const MotionItem = memo(({ label, isSelected, onPlay }: {
  label: string;
  isSelected: boolean;
  onPlay: () => void;
}) => (
  <Box
    {...sidebarStyles.historyDrawer.historyItem}
    {...(isSelected ? sidebarStyles.historyDrawer.historyItemSelected : {})}
    onClick={onPlay}
    mb={1}
    py={1.5}
    px={3}
    cursor="pointer"
  >
    <Text
      fontSize="sm"
      color={isSelected ? 'blue.200' : 'whiteAlpha.900'}
      fontWeight={isSelected ? 'semibold' : 'normal'}
    >
      {label}
    </Text>
  </Box>
));

MotionItem.displayName = 'MotionItem';

// Main component: a drawer listing every motion group/clip of the loaded
// model; clicking a clip force-plays it (priority 3) so it always wins over
// whatever idle/emotion motion is currently running.
function MotionDrawer({ children }: MotionDrawerProps): JSX.Element {
  const { t } = useTranslation();
  const [open, setOpen] = useState(false);
  const [groups, setGroups] = useState<MotionGroup[]>([]);
  // The clip the user last played, so the list can show what's selected.
  const [selectedKey, setSelectedKey] = useState<string | null>(null);

  // (Re)read the motion list each time the drawer opens — the model may have
  // loaded after mount, or the character may have been switched.
  useEffect(() => {
    if (open) setGroups(readMotionGroups());
  }, [open]);

  const playMotion = useCallback((group: string, index: number) => {
    const model = (window as any).getLive2DManager?.()?.getModel?.(0);
    model?.startMotion?.(group, index, 3); // 3 = LAppDefine.PriorityForce
    setSelectedKey(`${group}-${index}`);
  }, []);

  return (
    <DrawerRoot
      open={open}
      onOpenChange={(e) => setOpen(e.open)}
      placement="start"
    >
      <DrawerBackdrop />
      <DrawerTrigger asChild>{children}</DrawerTrigger>
      <DrawerContent style={sidebarStyles.historyDrawer.drawer.content}>
        <DrawerHeader>
          <DrawerTitle style={sidebarStyles.historyDrawer.drawer.title}>
            {t('motions.title')}
          </DrawerTitle>
          <DrawerCloseTrigger style={sidebarStyles.historyDrawer.drawer.closeButton} />
        </DrawerHeader>

        <DrawerBody>
          <Box {...sidebarStyles.historyDrawer.listContainer}>
            {groups.length === 0 && (
              <Text fontSize="sm" color="whiteAlpha.700">
                {t('motions.empty')}
              </Text>
            )}
            {groups.map((group) => (
              <Box key={group.name} mb={3}>
                <Text fontSize="xs" fontWeight="bold" textTransform="uppercase" color="blue.200" mb={1.5}>
                  {group.name}
                  {' '}
                  (
                  {group.clips.length}
                  )
                </Text>
                {group.clips.map((clip) => (
                  <MotionItem
                    key={`${group.name}-${clip.index}`}
                    label={clip.label}
                    isSelected={selectedKey === `${group.name}-${clip.index}`}
                    onPlay={() => playMotion(group.name, clip.index)}
                  />
                ))}
              </Box>
            ))}
          </Box>
        </DrawerBody>

        <DrawerFooter>
          <DrawerActionTrigger asChild>
            <Button {...sidebarStyles.historyDrawer.drawer.actionButton}>
              {t('common.close')}
            </Button>
          </DrawerActionTrigger>
        </DrawerFooter>
      </DrawerContent>
    </DrawerRoot>
  );
}

export default MotionDrawer;
